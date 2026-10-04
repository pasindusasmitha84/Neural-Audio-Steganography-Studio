# Neural Audio Steganography Studio

A single-file PyQt5 desktop application that hides secret text payloads inside audio using a lightweight PyTorch autoencoder operating in the frequency domain.

---

## 🔒 Key Features

* **Frequency-Domain Phase Coding:** Frames the carrier audio, computes an FFT per frame, and embeds payload bits into the *phase* of a key-selected set of mid-band bins. Because human hearing is largely insensitive to absolute phase, modifications sit below the psychoacoustic threshold, leaving the magnitude spectrogram visually identical to the original.
* **Zero-Training Deterministic Engine:** Relies on hard-coded, deterministic weights (secret bin permutation based on a secure key seed, phase levels, and magnitude floor) allowing it to work entirely out-of-the-box without any model training.
* **WAV Export Resilience:** Uses non-overlapping rectangular framing making the FFT/IFFT transform exactly invertible, ensuring that the hidden payload successfully survives standard WAV (int16) export and re-import cycles.
* **Interactive PyQt5 GUI & Spectrogram Studio:** Features real-time side-by-side magnitude spectrogram comparisons, objective SNR calculation, integrated audio playback, and seamless extract/save controls[cite: 7].

---

## 🛠️ Tech Stack

* **Core & Math:** Python[cite: 7], NumPy[cite: 7], SciPy[cite: 7]
* **Deep Learning:** PyTorch[cite: 7]
* **Audio Processing:** Librosa[cite: 7]
* **GUI & Visualization:** PyQt5[cite: 7], Matplotlib[cite: 7]

---

## 📦 Installation & Quick Start

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
   cd your-repo-name
