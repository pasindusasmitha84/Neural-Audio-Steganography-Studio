#!/usr/bin/env python3
"""
Neural Audio Steganography Studio
=================================
Single-file PyQt5 desktop application that hides a secret text payload inside
audio using a lightweight PyTorch autoencoder operating in the frequency domain.

Engine
------
* Encoder : frames the carrier, takes an FFT per frame and embeds payload bits in
            the *phase* of a key-selected set of mid-band bins (phase coding).
            The ear is largely insensitive to absolute phase, so the modification
            sits below the psychoacoustic threshold and the magnitude spectrogram
            of the covert audio is visually identical to the original.
* Decoder : mirrors the transform and reads the phase signs back.
* Weights : deterministic and hard-coded (secret bin permutation, phase levels,
            magnitude floor) -> works out-of-the-box, no training required.
* Framing : non-overlapping rectangular frames => FFT/IFFT is exactly invertible,
            so the payload survives WAV (int16) export and re-import.

Run  : python neural_audio_stego.py
Deps : pip install pyqt5 torch numpy scipy librosa matplotlib
"""

import math
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn
from scipy.io import wavfile
import librosa
import librosa.display

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtCore import QUrl
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtWidgets import (QApplication, QFileDialog, QGroupBox, QHBoxLayout,
                             QLabel, QMainWindow, QMessageBox, QPlainTextEdit,
                             QPushButton, QVBoxLayout, QWidget)

# --------------------------------------------------------------------------- #
#  Steganography engine (PyTorch)                                             #
# --------------------------------------------------------------------------- #
FRAME = 2048                 # samples per non-overlapping analysis frame
BAND_LO, BAND_HI = 24, 152   # usable FFT bins (~500 Hz - 3.3 kHz @ 44.1 kHz)
BITS_PER_FRAME = 64          # payload bits hidden in every frame
MAGIC = 0xA5                 # 8-bit sync marker in the header
HEADER_BITS = 24             # MAGIC (8) + payload length in bytes (16)
KEY_SEED = 1337              # secret key -> which bins carry data


class StegoAutoencoder(nn.Module):
    """Frequency-domain phase-coding autoencoder with frozen, pre-initialised weights."""

    def __init__(self):
        super().__init__()
        self.frame = FRAME
        self.bits_per_frame = BITS_PER_FRAME
        g = torch.Generator().manual_seed(KEY_SEED)
        perm = torch.randperm(BAND_HI - BAND_LO, generator=g)[:BITS_PER_FRAME] + BAND_LO
        # ---- hard-coded "weights" -------------------------------------------
        self.register_buffer("key_bins", perm)                                         # secret bin selection
        self.register_buffer("phase_levels", torch.tensor([-math.pi / 2, math.pi / 2]))  # bit 0 / bit 1
        self.mag_floor = nn.Parameter(torch.tensor(0.05), requires_grad=False)          # keeps bins above int16 noise
        self.eval()

    # ---- helpers -------------------------------------------------------------
    def _frames(self, x: torch.Tensor):
        n = (x.numel() // self.frame) * self.frame
        return x[:n].reshape(-1, self.frame), x[n:]

    def capacity_bytes(self, n_samples: int) -> int:
        bits = (n_samples // self.frame) * self.bits_per_frame - HEADER_BITS
        return max(0, min(bits // 8, 0xFFFF))

    @staticmethod
    def text_to_bits(text: str) -> torch.Tensor:
        payload = text.encode("utf-8")
        if len(payload) > 0xFFFF:
            raise ValueError("Payload too large (max 65535 bytes).")
        header = bytes([MAGIC, (len(payload) >> 8) & 0xFF, len(payload) & 0xFF])
        data = np.frombuffer(header + payload, dtype=np.uint8)
        return torch.from_numpy(np.unpackbits(data).copy())

    @staticmethod
    def bits_to_text(bits: torch.Tensor) -> str:
        if bits.numel() < HEADER_BITS:
            raise ValueError("Audio too short to contain a header.")
        arr = bits.to(torch.uint8).numpy()
        header = np.packbits(arr[:HEADER_BITS])
        if header[0] != MAGIC:
            raise ValueError("No payload found (sync marker mismatch or wrong key).")
        n_bytes = (int(header[1]) << 8) | int(header[2])
        need = HEADER_BITS + 8 * n_bytes
        if arr.size < need:
            raise ValueError("Audio truncated: payload declared longer than carrier.")
        return np.packbits(arr[HEADER_BITS:need]).tobytes().decode("utf-8", errors="replace")

    # ---- encoder -------------------------------------------------------------
    @torch.no_grad()
    def encode(self, audio: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        frames, tail = self._frames(audio)
        n_needed = math.ceil(bits.numel() / self.bits_per_frame)
        if n_needed > frames.shape[0]:
            raise ValueError(
                f"Payload needs {n_needed} frames but carrier only has {frames.shape[0]}. "
                "Use a longer carrier or a shorter message.")

        spec = torch.fft.rfft(frames)                 # (F, FRAME//2+1) complex
        mag, phase = spec.abs(), spec.angle()

        pad = n_needed * self.bits_per_frame - bits.numel()
        b = torch.cat([bits.long(), torch.zeros(pad, dtype=torch.long)])
        b = b.reshape(n_needed, self.bits_per_frame)

        # Phase coding: write bit -> phase level, keep magnitude (clamped to floor)
        rows = torch.arange(n_needed).unsqueeze(1)
        phase[rows, self.key_bins] = self.phase_levels[b]
        mag[rows, self.key_bins] = torch.clamp(mag[rows, self.key_bins], min=self.mag_floor)

        covert = torch.fft.irfft(torch.polar(mag, phase), n=self.frame)
        out = torch.cat([covert.reshape(-1), tail])
        return torch.clamp(out, -1.0, 1.0)

    # ---- decoder -------------------------------------------------------------
    @torch.no_grad()
    def decode(self, audio: torch.Tensor) -> torch.Tensor:
        frames, _ = self._frames(audio)
        spec = torch.fft.rfft(frames)
        imag = spec[:, self.key_bins].imag             # phase +pi/2 -> imag>0 -> bit 1
        return (imag > 0).reshape(-1)

    def forward(self, audio, bits):
        covert = self.encode(audio, bits)
        return covert, self.decode(covert)


# --------------------------------------------------------------------------- #
#  Audio helpers                                                              #
# --------------------------------------------------------------------------- #
def load_audio(path, sr=44100):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32), sr


def save_wav(path, y, sr):
    wavfile.write(path, sr, (np.clip(y, -1, 1) * 32767).astype(np.int16))


def demo_carrier(sr=44100, seconds=6.0):
    """Synth chord + soft noise so the app works with zero external files."""
    t = np.arange(int(sr * seconds)) / sr
    y = np.zeros_like(t)
    for f, a in [(220, .35), (277.18, .3), (329.63, .3), (440, .2), (659.25, .1)]:
        y += a * np.sin(2 * np.pi * f * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.5 * t + f))
    y += 0.01 * np.random.default_rng(7).standard_normal(t.size)
    return (0.8 * y / np.abs(y).max()).astype(np.float32), sr


# --------------------------------------------------------------------------- #
#  GUI                                                                        #
# --------------------------------------------------------------------------- #
class SpectrogramCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(12, 4.5), dpi=110, tight_layout=True)
        super().__init__(self.fig)
        self.ax1 = self.fig.add_subplot(1, 2, 1)
        self.ax2 = self.fig.add_subplot(1, 2, 2)

    def plot(self, original, covert, sr):
        for ax, y, title in ((self.ax1, original, "Original Audio"),
                             (self.ax2, covert, "Covert Audio (payload embedded)")):
            ax.clear()
            if y is None:
                ax.set_title(title + " — nothing loaded")
                continue
            S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=2048, hop_length=256)), ref=np.max)
            librosa.display.specshow(S, sr=sr, hop_length=256, x_axis="time", y_axis="log",
                                     ax=ax, cmap="magma", vmin=-80, vmax=0)
            ax.set_title(title)
        if original is not None and covert is not None:
            n = min(original.size, covert.size)
            snr = 10 * np.log10(np.sum(original[:n] ** 2) / (np.sum((original[:n] - covert[:n]) ** 2) + 1e-12))
            self.fig.suptitle(f"Magnitude spectrograms (dB) — SNR original vs covert: {snr:.1f} dB")
        self.draw()


class StegoApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Audio Steganography Studio")
        self.resize(1400, 820)
        self.model = StegoAutoencoder()
        self.sr = 44100
        self.original = None
        self.covert = None
        self.covert_path = os.path.join(tempfile.gettempdir(), "covert_audio.wav")
        self.player = QMediaPlayer()
        self._build_ui()
        self.on_demo()

    # ---- layout ----------------------------------------------------------------
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        # carrier controls
        carrier_box = QGroupBox("1. Carrier audio")
        cl = QHBoxLayout(carrier_box)
        self.btn_load = QPushButton("Load Audio…")
        self.btn_demo = QPushButton("Generate Demo Carrier")
        self.lbl_info = QLabel("")
        self.btn_load.clicked.connect(self.on_load)
        self.btn_demo.clicked.connect(self.on_demo)
        cl.addWidget(self.btn_load)
        cl.addWidget(self.btn_demo)
        cl.addWidget(self.lbl_info, 1)
        layout.addWidget(carrier_box)

        # payload controls
        payload_box = QGroupBox("2. Secret Payload")
        pl = QVBoxLayout(payload_box)
        self.txt_payload = QPlainTextEdit()
        self.txt_payload.setPlaceholderText("Type the secret message to hide…")
        self.txt_payload.setPlainText("Meet at the old lighthouse, 02:00. Bring the drive.")
        self.txt_payload.setMaximumHeight(90)
        pl.addWidget(self.txt_payload)
        bl = QHBoxLayout()
        self.btn_encode = QPushButton("▶  Encode & Play")
        self.btn_extract = QPushButton("🔍  Extract Payload")
        self.btn_save = QPushButton("💾  Save Covert WAV…")
        self.btn_open_covert = QPushButton("📂  Open Covert WAV & Extract…")
        self.btn_encode.clicked.connect(self.on_encode)
        self.btn_extract.clicked.connect(self.on_extract)
        self.btn_save.clicked.connect(self.on_save)
        self.btn_open_covert.clicked.connect(self.on_open_covert)
        for b in (self.btn_encode, self.btn_extract, self.btn_save, self.btn_open_covert):
            bl.addWidget(b)
        pl.addLayout(bl)
        layout.addWidget(payload_box)

        # spectrograms
        spec_box = QGroupBox("3. Spectrogram comparison")
        sl = QVBoxLayout(spec_box)
        self.canvas = SpectrogramCanvas()
        sl.addWidget(self.canvas)
        layout.addWidget(spec_box, 1)

        self.status = QLabel("Ready.")
        self.status.setStyleSheet("padding:4px; color:#2b7a0b; font-weight:bold;")
        layout.addWidget(self.status)

    # ---- helpers ---------------------------------------------------------------
    def _set_carrier(self, y, sr, source):
        self.original, self.sr, self.covert = y, sr, None
        cap = self.model.capacity_bytes(y.size)
        self.lbl_info.setText(f"{source} — {y.size / sr:.1f}s @ {sr} Hz — capacity ≈ {cap:,} bytes "
                              f"({BITS_PER_FRAME} bits / {FRAME}-sample frame)")
        self.canvas.plot(self.original, None, sr)
        self.status.setText("Carrier loaded. Enter a payload and press Encode & Play.")

    def _error(self, msg):
        self.status.setText("Error: " + msg)
        QMessageBox.critical(self, "Steganography error", msg)

    # ---- actions ---------------------------------------------------------------
    def on_demo(self):
        y, sr = demo_carrier()
        self._set_carrier(y, sr, "Synthetic demo carrier")

    def on_load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open carrier audio", "",
                                              "Audio (*.wav *.flac *.mp3 *.ogg *.m4a);;All files (*)")
        if not path:
            return
        try:
            y, sr = load_audio(path)
            self._set_carrier(y, sr, os.path.basename(path))
        except Exception as e:
            self._error(f"Could not load audio: {e}")

    def on_encode(self):
        if self.original is None:
            return self._error("Load or generate a carrier first.")
        text = self.txt_payload.toPlainText()
        if not text:
            return self._error("Secret payload is empty.")
        try:
            bits = self.model.text_to_bits(text)
            covert = self.model.encode(torch.from_numpy(self.original), bits)
            self.covert = covert.numpy().astype(np.float32)
            save_wav(self.covert_path, self.covert, self.sr)
            self.canvas.plot(self.original, self.covert, self.sr)
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.covert_path)))
            self.player.play()
            self.status.setText(f"Embedded {len(text.encode('utf-8'))} bytes ({bits.numel()} bits incl. header) "
                                f"in {math.ceil(bits.numel() / BITS_PER_FRAME)} frames. Playing covert audio.")
        except Exception as e:
            self._error(str(e))

    def _extract_from(self, y, label):
        try:
            text = self.model.bits_to_text(self.model.decode(torch.from_numpy(y.astype(np.float32))))
            self.status.setText(f"Payload extracted from {label}.")
            QMessageBox.information(self, "Extracted payload", text)
        except Exception as e:
            self._error(str(e))

    def on_extract(self):
        if self.covert is None:
            return self._error("No covert audio yet — press Encode & Play first.")
        # Re-read the exported int16 WAV to prove the payload survives a real file round-trip.
        y, _ = load_audio(self.covert_path, sr=self.sr)
        self._extract_from(y, "covert WAV (int16 round-trip)")

    def on_save(self):
        if self.covert is None:
            return self._error("No covert audio to save.")
        path, _ = QFileDialog.getSaveFileName(self, "Save covert WAV", "covert.wav", "WAV (*.wav)")
        if path:
            save_wav(path, self.covert, self.sr)
            self.status.setText(f"Saved covert audio to {path}")

    def on_open_covert(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open covert WAV", "", "WAV (*.wav)")
        if not path:
            return
        try:
            y, sr = load_audio(path, sr=None)
            self.covert, self.sr = y, sr
            self.canvas.plot(self.original, self.covert, sr)
            self._extract_from(y, os.path.basename(path))
        except Exception as e:
            self._error(f"Could not read file: {e}")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = StegoApp()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
