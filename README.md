# SignDETR

SignDETR is a DETR (DEtection TRansformer) based real-time sign language detection project using PyTorch, OpenCV, and Albumentations. It detects sign language gestures (`hello`, `iloveyou`, `thankyou`) directly from your webcam.

## What It Does

- Captures frames from your webcam in real-time.
- Runs DETR (ResNet-50 backbone + Transformer encoder/decoder) inference on captured frames.
- Predicts bounding boxes, classes, and confidence scores.
- Displays bounding boxes, confidence badges, and detection stats in an OpenCV window.

## Setup

```powershell
pip install uv
uv sync
```

## Running The Apps

### 1. Streamlit Web Frontend (Recommended)

To launch the web interface with live streaming, browser snapshot, and image upload:

```powershell
uv run streamlit run streamlit_app.py
```

### 2. OpenCV Desktop Window

To run the direct OpenCV webcam window:

```powershell
uv run src/realtime.py
```

If multiple cameras are connected, specify your webcam index:

```powershell
$env:SIGNDETR_CAMERA_INDEX="0"
uv run src/realtime.py
```

Press **q** in the OpenCV window to exit.

## Model & Classes

- Checkpoint: `pretrained/4426_model.pt`
- Detection Classes:
  - `hello`
  - `iloveyou`
  - `thankyou`

## Repository Structure

```
├── data/              # Original training and test datasets
│   ├── train/
│   └── test/
├── pretrained/        # Model checkpoints
│   └── 4426_model.pt  # Original pretrained DETR checkpoint
├── src/
│   ├── model.py       # DETR architecture definition
│   ├── loss.py        # Hungarian matcher & SetCriterion loss
│   ├── data.py        # Dataset & DataLoader pipeline
│   ├── train.py       # Training script
│   ├── test.py        # Evaluation script
│   ├── realtime.py    # Real-time OpenCV webcam detection script
│   ├── config.json    # Target class labels and display colors
│   └── utils/         # Camera, box rescaling, logger, and display helpers
├── Cheatsheet.png     # Reference gestures
├── pyproject.toml     # Project dependencies
└── uv.lock
```

## Author

Original project by Nick Renotte.