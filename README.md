# 🧠 Real-Time AI Handwriting Assistant with Voice-Based Spelling Correction

A real-time AI system that captures handwritten text from a live camera feed, converts it into digital text using deep learning–based OCR, performs spelling correction, and speaks the corrected text aloud.

This project combines **computer vision**, **transformer-based OCR**, **NLP spell correction**, and **text-to-speech**, forming a complete perception-to-action pipeline.

---

## 🚀 Features

- 📷 Real-time webcam capture
- ✍️ Handwritten text recognition using **TrOCR (Transformer OCR)**
- 📝 Automatic spelling correction
- 🔊 Voice output of corrected text
- 🧠 Motion-triggered processing (efficient & intelligent)
- ⚡ Runs on CPU or GPU

---

## 🏗️ System Architecture

Camera Feed
↓
Motion Detection (OpenCV)
↓
Handwritten OCR (TrOCR)
↓
Spell Correction (pyspellchecker)
↓
Text-to-Speech (pyttsx3)

---

## 🧪 Tech Stack

| Component | Technology |
|--------|-----------|
| OCR | HuggingFace TrOCR |
| Vision | OpenCV |
| NLP | pyspellchecker |
| Speech | pyttsx3 |
| ML Framework | PyTorch |
| Image Processing | PIL |
| Hardware | Webcam (CPU/GPU supported) |

---

## 📂 Project Structure

SPELL_CORRECTOR_LLM/
│
├── handwritten_recogonitation/
│ ├── Models/
│ ├── configs.py
│ ├── model.py
│ ├── inferenceModel.py
│ └── train.py
│
├── corrector.py # Real-time OCR + correction + voice
├── README.md
├── .gitignore
└── .venv/ # (ignored)


> ❗ Datasets are intentionally excluded from GitHub due to size constraints.

---

## 📦 Dataset (Not Included)

This project was trained using the **IAM Handwriting Dataset**.

🔗 Download manually from:
https://fki.tic.heia-fr.ch/databases/iam-handwriting-database

Place datasets locally if you plan to retrain models.

---

## ⚙️ Installation

### 1️⃣ Clone the repository
git clone https://github.com/gowthamx25/real-time-ai-handwriting-assistant.git
cd real-time-ai-handwriting-assistant
2️⃣ Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux / Mac
.venv\Scripts\activate      # Windows
3️⃣ Install dependencies
pip install torch torchvision torchaudio
pip install opencv-python pyttsx3 pillow numpy
pip install transformers pyspellchecker
▶️ Run the Application
python corrector.py
Controls
q → Quit application

Speak handwritten text automatically when motion is detected

🧠 Key Highlights
Uses Transformer-based OCR, not traditional Tesseract

Motion-based triggering reduces unnecessary computation

Modular design — easy to extend with LLMs or cloud APIs

Fully offline pipeline (no API cost)

🔮 Future Improvements
Multilingual handwriting recognition

Grammar correction using LLMs

Mobile camera support

Web-based dashboard

Sentence-level language correction

👨‍💻 Author
Gowtham S
AI & Data Science Student | Aspiring MLOps Engineer

🔗 GitHub: https://github.com/gowthamx25

⭐ If you find this useful
Give the repository a ⭐ — it helps a lot!
