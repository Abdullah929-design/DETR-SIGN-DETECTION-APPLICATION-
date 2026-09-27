# 🤟 SignDETR: Real-Time Sign Language Detection with DETR

<img width="2560" height="1600" alt="detr-sign-detection streamlit app(Nest Hub Max)" src="https://github.com/user-attachments/assets/76305f4b-962e-404a-90fc-eb48ea8e8eaf" />


[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![WebRTC](https://img.shields.io/badge/WebRTC-Live_Streaming-333333?style=for-the-badge&logo=webrtc&logoColor=white)](https://webrtc.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**SignDETR** is an end-to-end computer vision and deep learning project that brings **DEtection TRansformer (DETR)** to sign language gesture recognition. Utilizing a ResNet-50 convolutional backbone combined with a Transformer encoder-decoder architecture, SignDETR detects American Sign Language (ASL) gestures in real time with high accuracy and bounding box localization.

The project features a **Streamlit Web Studio** supporting browser-based WebRTC live camera streaming, snapshot inference, and image uploads, alongside the original high-performance **OpenCV desktop viewer**.

---

## 🌐 Live Web Demo

Access the live cloud deployment directly in your web browser:  
👉 **[Launch SignDETR Web Studio](https://detr-sign-detection.streamlit.app/)**

---

## 📸 Key Features

### 1. 📹 Interactive Streamlit Web Studio
* **Live WebRTC Browser Stream:** Peer-to-peer real-time video streaming directly through web browsers (Chrome, Edge, Firefox, Mobile Safari) with continuous bounding box overlays and detection labels.
* **Browser Camera Snapshot:** Take a photo directly in the browser to receive an instant bounding box breakdown with confidence scores.
* **Image File Upload:** Test detection on arbitrary `.jpg`, `.jpeg`, or `.png` images without requiring a webcam.
* **Real-Time Sensitivity Tuning:** Dynamically adjust the confidence threshold slider (0.10 to 0.95) and toggle horizontal mirroring.

### 2. 🖥️ Native OpenCV Desktop Application
* High-framerate desktop camera inference using DirectShow (`CAP_DSHOW`) on Windows.
* Rich terminal telemetry tracking inference latency (ms), frame rates (FPS), and bounding box coordinates.

### 3. ☁️ Zero-Friction Cloud Deployment
* **Automated Weight Provisioning:** The app automatically retrieves the ~108 MB pretrained PyTorch weights from the CDN on cold startup, eliminating Git LFS storage bottlenecks and manual downloads.
* **Headless Linux Ready:** Packaged with `requirements.txt` and `packages.txt` for 1-click deployment on Streamlit Community Cloud.

---

## 🎯 Target Gesture Classes

| Sign | Emoji | Class Label | Description |
| :--- | :---: | :---: | :--- |
| **Hello** | 🖐️ | `hello` | Open palm with fingers extended facing the camera. |
| **I Love You** | 🤟 | `iloveyou` | Thumb, index, and pinky fingers extended; middle and ring fingers down. |
| **Thank You** | 🙏 | `thankyou` | Flat hand moving forward from chin/chest towards the camera. |

> A visual reference guide is available in [Cheatsheet.png](Cheatsheet.png) and expandable within the web studio sidebar.

---

## 🧠 Model Architecture & Technical Details

```
Input Frame (224x224x3)
         │
         ▼
┌─────────────────────────┐
│   ResNet-50 Backbone    │  <-- Feature Extraction (ImageNet Pretrained)
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│   1x1 Convolution       │  <-- Reduces channel dimension to 256 (hidden_dim)
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Transformer Encoder    │  <-- 1 Layer, 8 Multi-Head Self-Attention
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Transformer Decoder    │  <-- 1 Layer, 25 Learned Object Queries
└───────────┬─────────────┘
            │
      ┌─────┴─────────────────────┐
      ▼                           ▼
┌──────────────┐          ┌──────────────┐
│ Class Head   │          │  BBox Head   │
│ (Linear)     │          │  (3-layer MLP│
└──────────────┘          └──────────────┘
      │                           │
      ▼                           ▼
Class Probabilities      Normalized (cx, cy, w, h)
```

* **Backbone:** ResNet-50 with frozen early stages for robust visual feature extraction.
* **Positional Embeddings:** 2D sinusoidal spatial positional encodings.
* **Object Queries:** 25 learned query slots evaluated in parallel.
* **Bipartite Matching Loss:** Trained using Hungarian algorithm matching with generalized box IoU (`GIoU`), L1 coordinate distance, and classification cross-entropy via `SetCriterion`.
* **Preprocessing:** Normalized with ImageNet statistics (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`) and resized to `224×224`.

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Deep Learning** | PyTorch, Torchvision | Model definition, tensor operations, ResNet-50 backbone |
| **Detection Transformer** | DETR Architecture | Attention-based end-to-end object detection |
| **Web Framework** | Streamlit (v1.37+) | Interactive web UI, sidebar controls, dashboard layout |
| **Web Streaming** | Streamlit-WebRTC, PyAV | Low-latency WebRTC browser video streaming pipeline |
| **Computer Vision** | OpenCV (cv2) | Frame capture, bounding box rendering, video I/O |
| **Image Preprocessing** | Albumentations | Fast resize, ImageNet normalization, PyTorch tensor conversion |
| **Environment & Package Mgmt** | `uv`, setuptools, pip | Fast dependency resolution and environment isolation |
| **Cloud Hosting** | Streamlit Community Cloud | Free, serverless hosting directly connected to GitHub |

---

## 🚀 Local Installation & Setup

### Prerequisites
* Python 3.11 or higher
* Webcam (built-in or USB external)
* [uv](https://docs.astral.sh/uv/) (recommended) or standard `pip`

### Step 1: Clone the Repository
```powershell
git clone https://github.com/Abdullah929-design/DETR-SIGN-DETECTION-APPLICATION-.git
cd DETR-SIGN-DETECTION-APPLICATION-
```

### Step 2: Install Dependencies

#### Using `uv` (Fastest):
```powershell
pip install uv
uv sync
```

#### Using standard `pip`:
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 💻 Running the Applications

### 1. Launch the Streamlit Web Studio (Recommended)
```powershell
uv run streamlit run streamlit_app.py
```
*Or with active virtual environment:*
```powershell
streamlit run streamlit_app.py
```
The browser will automatically open at `http://localhost:8501`.

### 2. Launch the Native OpenCV Desktop Viewer
```powershell
uv run src/realtime.py
```
* If you have multiple webcams connected, specify the index before launching:
```powershell
$env:SIGNDETR_CAMERA_INDEX="0"
uv run src/realtime.py
```
* Press **`q`** in the video window to quit.

---

## ☁️ Cloud Deployment Guide (Streamlit Community Cloud)

This repository is pre-configured with `requirements.txt` and `packages.txt` for 1-click deployment on Streamlit Community Cloud:

1. Fork or push this repository to your GitHub account.
2. Navigate to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **"New App"** and enter:
   * **Repository:** `Abdullah929-design/DETR-SIGN-DETECTION-APPLICATION-`
   * **Branch:** `main`
   * **Main file path:** `streamlit_app.py`
4. Click **"Deploy!"**.
5. Streamlit Cloud will install system libraries (`libgl1`), Python dependencies, download the model weights automatically from the CDN on startup, and launch your live URL.

---

## 📁 Repository Structure

```
DETR-SIGN-DETECTION-APPLICATION-/
│
├── .gitignore               # Ignored build artifacts, checkpoints, and caches
├── Cheatsheet.png           # Visual reference for sign gestures
├── packages.txt             # System OpenGL/GLib packages for cloud Linux deployment
├── pyproject.toml           # Project metadata and dependencies (uv / pip)
├── README.md                # Project documentation
├── requirements.txt         # Pinned requirements for Streamlit Cloud deployment
├── streamlit_app.py         # Streamlit Web Studio (WebRTC, Snapshot & Upload)
├── uv.lock                  # Lockfile ensuring reproducible environment
│
├── data/                    # Dataset directory
│   ├── test/                # Test split images and YOLO-style labels
│   └── train/               # Train split images and YOLO-style labels
│
├── pretrained/              # Local model weights directory
│   └── 4426_model.pt        # Pretrained DETR PyTorch model checkpoint (108 MB)
│
└── src/                     # Core source code
    ├── __init__.py
    ├── config.json          # Target sign categories and bounding box colors
    ├── data.py              # Custom PyTorch Dataset with Albumentations pipeline
    ├── loss.py              # Hungarian Matcher and SetCriterion loss implementation
    ├── model.py             # DETR PyTorch nn.Module architecture definition
    ├── realtime.py          # Native OpenCV real-time desktop webcam detector
    ├── test.py              # Model validation and evaluation script
    ├── train.py             # Model training script
    └── utils/               # Utilities and helper modules
        ├── boxes.py         # Coordinate conversions (cxcywh <-> xyxy) and IoU
        ├── camera.py        # Safe camera initialization (DirectShow / MSMF)
        ├── collect_images.py# Webcam dataset collection helper
        ├── display.py       # Fullscreen window helpers
        ├── linearsumeg.py   # Bipartite assignment solver
        ├── logger.py        # Structured logging utilities
        ├── rich_handlers.py # Terminal tables and performance formatters
        ├── setup.py         # Config reader for classes and color mapping
        └── testprogress.py  # Progress tracking helpers
```

---

## 🤝 Credits & Acknowledgements

* **Original Project Base:** Created by [Nick Renotte](https://github.com/nicknochnack/SignDETR).
* **Enhanced & Maintained by:** [Abdullah](https://github.com/Abdullah929-design).
* **Reference Paper:** Carion et al., *"End-to-End Object Detection with Transformers"* (ECCV 2020) — [Paper](https://arxiv.org/abs/2005.12872).

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
