# 🎨 AI Colorist

> **Professional photo and video color grading driven by plain-English commands without generative repainting.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Android](https://img.shields.io/badge/Platform-Android%20Compose-3DDC84.svg)](https://developer.android.com/jetpack/compose)

AI Colorist allows photographers and videographers to color grade photos and videos by simply typing what they want (e.g. *"Warm golden hour glow, make the sky deeper blue, protect skin tones"*).

Unlike diffusion image-generation models that hallucinate and alter facial features or textures, **AI Colorist applies real parametric grading operations** (linear exposure stops, curves, 3-way color wheels, HSL shifts) directly onto the original pixels.

---

## ✨ Features

- **Linear-Light Exposure & Contrast**: Exposure adjustments in linear gamma ($2^{\text{stops}}$) and PCHIP interpolator S-curves.
- **Color Wheels & HSL**: 3-way Lift/Gamma/Gain split wheels (Shadows, Midtones, Highlights) and 8-color targeted HSL shifting.
- **Colorist Guardrails**:
  - **Skin-Tone Protection**: Automatic $YCrCb$ chrominance shielding to keep natural skin tones unharmed during heavy creative grades.
  - **Clipping Protection**: Dynamic highlight/shadow safety checks preventing crushed blacks or blown-out whites.
- **Classic & AI Masking**:
  - *Classic*: Luminance bands (shadows/midtones/highlights), color isolation, linear gradients, and soft vignettes.
  - *AI Objects*: Grounding DINO + SAM (Segment Anything) to isolate *"sky"*, *"person"*, *"car"*, etc., with point-refinement (`add`/`remove` spots).
- **Streaming Video Pipeline**: Memory-efficient FFmpeg raw pipe streaming, automated scene cut detection, per-shot grading, and optical flow mask tracking across video frames.
- **REST API Backend**: Production-ready FastAPI service with asynchronous task processing for video.
- **Android Mobile App**: Modern Jetpack Compose interface with an interactive real-time **Before / After split slider**.

---

## 🏗️ Architecture

```
User Prompt ("Warm sunset, moody shadows")
                 │
                 ▼
      ┌─────────────────────┐
      │  Gemini Brain (LLM) │ ──► Reads photo stats (luma, warmth, colorfulness)
      └─────────────────────┘
                 │
                 ▼ (Outputs JSON Plan)
      ┌─────────────────────┐
      │  Colorist Engine    │ ──► Applies Linear Exposure, Curves, Wheels & HSL
      └─────────────────────┘
                 │
                 ▼
      ┌─────────────────────┐
      │  Guardrails Check   │ ──► Skin protection & Highlight/Shadow clipping check
      └─────────────────────┘
                 │
                 ▼
         Graded Result (Zero Pixel Hallucination)
```

---

## 📁 Repository Structure

```
ai-colorist/
├── colorist.py              # Core grading engine (dials, curves, wheels, sessions)
├── brain.py                 # Gemini API integration & JSON plan generator
├── masks.py                 # Classic region masks (luminance, color, gradients)
├── aimasks.py               # AI object masks (Grounding DINO + SAM) & refinement
├── video.py                 # Video reading/writing, scene detection & mask tracking
├── server.py                # FastAPI backend API for mobile & cloud
├── Dockerfile               # Production container with FFmpeg & PyTorch
├── requirements.txt         # Core dependencies
├── ai_colorist_master.ipynb # Complete step-by-step master notebook
└── android_project/         # Native Android application (Jetpack Compose)
    ├── app/src/main/
    │   ├── AndroidManifest.xml
    │   └── java/com/aicolorist/
    │       ├── MainActivity.kt
    │       ├── data/ColoristApiService.kt
    │       └── ui/
    │           ├── ColoristScreen.kt     # Interactive split-view UI
    │           └── ColoristViewModel.kt
    └── build.gradle.kts
```

---

## 🚀 Quickstart

### 1. In Google Colab
Open [`ai_colorist_master.ipynb`](ai_colorist_master.ipynb) in Google Colab to run the complete interactive pipeline cell-by-cell.

### 2. Run the FastAPI Server Locally
```bash
# Clone the repository
git clone https://github.com/Prayansh-builds/ai-colorist.git
cd ai-colorist

# Install dependencies
pip install -r requirements.txt
pip install fastapi uvicorn[standard] python-multipart

# Set your Gemini API Key and run
export GEMINI_API_KEY="your-api-key"
uvicorn server:app --host 0.0.0.0 --port 8000
```

Interactive API documentation will be available at `http://localhost:8000/docs`.

### 3. Deploy with Docker
```bash
docker build -t ai-colorist .
docker run -p 8000:8000 -e GEMINI_API_KEY="your-key" ai-colorist
```

---

## 📱 Android App

The Android client is built with **Jetpack Compose**, featuring:
- **Zero-Permission Photo Picker** using modern Android 14+ contracts.
- **Interactive Drag-to-Compare Split Slider**.
- Direct connection to your self-hosted FastAPI server.

To build the APK:
1. Open the `android_project` folder in **Android Studio**.
2. Click **Build $\to$ Build Bundle(s) / APK(s) $\to$ Build APK(s)**.

---

## 🗺️ Roadmap & Status

- [x] **Part 1: Core Engine** (Exposure, curves, color wheels, HSL, skin protection)
- [x] **Part 2: Masking Engine** (Classic luminance/color masks + AI Grounding DINO & SAM)
- [x] **Part 3: Video Engine** (FFmpeg streaming, scene cut detection, per-shot grading)
- [x] **Part 4: Server API** (FastAPI backend + Docker containerization)
- [x] **Part 5: Android Mobile App** (Compose UI, before/after split slider, Retrofit client)
- [ ] **Part 6: Google Play Store Release** (In Progress)

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
