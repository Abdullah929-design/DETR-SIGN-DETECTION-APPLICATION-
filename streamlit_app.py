from __future__ import annotations

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

# Suppress rich logging and ensure utf-8 encoding for Windows cp1252 safety
os.environ["USE_RICH_LOGGING"] = "0"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import cv2
import numpy as np
import streamlit as st
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

# WebRTC imports for browser-based live video streaming
try:
    import av
    from streamlit_webrtc import (
        webrtc_streamer,
        VideoProcessorBase,
        RTCConfiguration,
        WebRtcMode,
    )
    HAS_WEBRTC = True
except ImportError:
    HAS_WEBRTC = False

# Ensure project root and src directory are in sys.path
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
for p in (str(SRC_DIR), str(ROOT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from model import DETR
from utils.boxes import rescale_bboxes
from utils.setup import get_classes, get_colors
from utils.camera import open_default_camera

# ---------------------------------------------------------
# Streamlit Page Config & Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="SignDETR Studio",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 1.25rem 1.75rem;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .main-header p {
        margin: 0.3rem 0 0 0;
        color: #94a3b8;
        font-size: 0.95rem;
    }
    .stat-badge {
        display: inline-block;
        padding: 0.3rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
        background-color: #0f766e;
        color: #ccfbf1;
        border: 1px solid #14b8a6;
    }
    .class-card {
        background: #1e293b;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        border: 1px solid #334155;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .class-icon {
        font-size: 1.4rem;
    }
    .class-label {
        font-weight: 700;
        color: #38bdf8;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Cached Model & Transform
# ---------------------------------------------------------
@st.cache_resource
def load_detr_model(checkpoint_path: str = "pretrained/4426_model.pt", num_classes: int = 3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DETR(num_classes=num_classes)
    model.to(device)
    model.eval()

    full_path = ROOT_DIR / checkpoint_path
    # Automatically download weights if missing or if file is an LFS text pointer (< 1MB)
    if not full_path.exists() or full_path.stat().st_size < 1_000_000:
        full_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/nicknochnack/SignDETR/raw/main/pretrained/4426_model.pt"
        with st.spinner("Downloading pretrained DETR model weights (~108 MB)..."):
            import urllib.request
            urllib.request.urlretrieve(url, str(full_path))

    model.load_pretrained(str(full_path))
    return model, device

@st.cache_resource
def get_transform():
    return A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

CLASSES = get_classes()
COLORS = get_colors()

# ---------------------------------------------------------
# Detection Pipeline
# ---------------------------------------------------------
def run_detection(
    image_bgr: np.ndarray,
    model: DETR,
    device: torch.device,
    transform: A.Compose,
    threshold: float = 0.45,
) -> Tuple[np.ndarray, List[Dict], Dict]:
    orig_h, orig_w = image_bgr.shape[:2]

    transformed = transform(image=image_bgr)
    image_tensor = transformed["image"].unsqueeze(0).to(device)

    start_time = time.time()
    with torch.no_grad():
        result = model(image_tensor)
    latency_ms = (time.time() - start_time) * 1000

    # Softmax excluding the no-object slot (last column)
    probabilities = result["pred_logits"].softmax(-1)[:, :, :-1]
    max_probs, max_classes = probabilities.max(-1)

    # Top overall query across the 25 object queries
    top_prob, top_q_idx = max_probs[0].max(0)
    top_c_idx = max_classes[0, top_q_idx].item()
    top_prediction = {
        "class": CLASSES[top_c_idx] if top_c_idx < len(CLASSES) else f"class_{top_c_idx}",
        "confidence": float(top_prob.item()),
    }

    # Filter by user threshold
    keep_mask = max_probs > threshold
    batch_indices, query_indices = torch.where(keep_mask)

    detections = []
    annotated_frame = image_bgr.copy()

    if len(batch_indices) > 0:
        bboxes = rescale_bboxes(
            result["pred_boxes"][batch_indices, query_indices, :],
            (orig_w, orig_h),
        )
        classes = max_classes[batch_indices, query_indices]
        probas = max_probs[batch_indices, query_indices]

        for bclass, bprob, bbox in zip(classes, probas, bboxes):
            c_idx = int(bclass.cpu().item())
            prob_val = float(bprob.cpu().item())
            x1, y1, x2, y2 = [float(v.cpu().item()) for v in bbox]

            x1 = max(0, min(orig_w - 1, x1))
            y1 = max(0, min(orig_h - 1, y1))
            x2 = max(0, min(orig_w - 1, x2))
            y2 = max(0, min(orig_h - 1, y2))

            class_name = CLASSES[c_idx] if c_idx < len(CLASSES) else f"class_{c_idx}"
            color = COLORS[c_idx % len(COLORS)]

            detections.append(
                {
                    "class": class_name,
                    "confidence": prob_val,
                    "bbox": (round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)),
                    "latency_ms": latency_ms,
                }
            )

            box_color = (int(color[0]), int(color[1]), int(color[2]))
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 3)

            label = f"{class_name.upper()} {prob_val * 100:.1f}%"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1
            )
            top_y = max(int(y1) - 8, label_h + 8)

            cv2.rectangle(
                annotated_frame,
                (int(x1), top_y - label_h - 6),
                (int(x1) + label_w + 10, top_y + baseline),
                box_color,
                -1,
            )
            cv2.putText(
                annotated_frame,
                label,
                (int(x1) + 5, top_y - 2),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return annotated_frame, detections, top_prediction


# ---------------------------------------------------------
# WebRTC Video Processor for Real-time Browser Stream
# ---------------------------------------------------------
RTC_CONFIG = (
    RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    if HAS_WEBRTC
    else None
)

if HAS_WEBRTC:
    class SignDetectionVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.threshold = 0.45
            self.mirror = False
            self.model = None
            self.device = None
            self.transform = None

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img_bgr = frame.to_ndarray(format="bgr24")
            if self.mirror:
                img_bgr = cv2.flip(img_bgr, 1)

            if self.model is not None and self.transform is not None:
                annotated_bgr, _, _ = run_detection(
                    img_bgr, self.model, self.device, self.transform, threshold=self.threshold
                )
            else:
                annotated_bgr = img_bgr

            return av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Detection Settings")

    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.45,
        step=0.05,
        help="Detection threshold. Set lower (e.g. 0.35-0.45) if detections are not showing.",
    )

    mirror_input = st.checkbox("Mirror Camera / Flip Horizontal", value=False)

    st.markdown("---")
    st.markdown("### 🎯 Detectable Hand Signs")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(
            '<div class="class-card"><span class="class-icon">🖐️</span><br><span class="class-label">hello</span></div>',
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            '<div class="class-card"><span class="class-icon">🤟</span><br><span class="class-label">iloveyou</span></div>',
            unsafe_allow_html=True,
        )
    with col_c:
        st.markdown(
            '<div class="class-card"><span class="class-icon">🙏</span><br><span class="class-label">thankyou</span></div>',
            unsafe_allow_html=True,
        )

    cheatsheet_path = ROOT_DIR / "Cheatsheet.png"
    if cheatsheet_path.exists():
        with st.expander("📖 Reference Cheatsheet"):
            st.image(str(cheatsheet_path), caption="Reference Gestures", width="stretch")

    st.markdown("---")
    if st.button("🖥️ Open Desktop Window (OpenCV)"):
        st.info("Launching desktop OpenCV window in background...")
        subprocess.Popen([sys.executable, "src/realtime.py"], cwd=str(ROOT_DIR))


# ---------------------------------------------------------
# Main UI Header
# ---------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>🤟 SignDETR Sign Language Studio</h1>
        <p>Real-time American Sign Language detection powered by DETR Transformer & PyTorch.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load model
try:
    with st.spinner("Loading DETR model checkpoint..."):
        model, device = load_detr_model("pretrained/4426_model.pt", num_classes=3)
        transform = get_transform()
    st.markdown(
        f'<span class="stat-badge">✅ DETR Model Ready</span>'
        f'<span class="stat-badge">Device: {device.type.upper()}</span>'
        f'<span class="stat-badge">Classes: {len(CLASSES)} ({", ".join(CLASSES)})</span>'
        f'<span class="stat-badge">Threshold: {int(confidence_threshold * 100)}%</span>',
        unsafe_allow_html=True,
    )
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Mode Selection
# ---------------------------------------------------------
available_modes = []
if HAS_WEBRTC:
    available_modes.append("📹 Live Browser Video (WebRTC)")
available_modes.extend(["📸 Browser Camera Snap", "📁 Upload Image"])

mode = st.radio(
    "Choose Input Mode:",
    available_modes,
    index=0,
    horizontal=True,
)

st.markdown("---")

# ----------------- Mode 1: Live Browser Video (WebRTC) -----------------
if HAS_WEBRTC and mode == "📹 Live Browser Video (WebRTC)":
    st.markdown("#### Real-time Browser Webcam Stream (WebRTC)")
    st.caption("Streams your webcam continuously directly inside the web browser with real-time DETR sign language detection.")

    webrtc_ctx = webrtc_streamer(
        key="sign-detr-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=SignDetectionVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.threshold = confidence_threshold
        webrtc_ctx.video_processor.mirror = mirror_input
        webrtc_ctx.video_processor.model = model
        webrtc_ctx.video_processor.device = device
        webrtc_ctx.video_processor.transform = transform

    st.markdown(
        """
        > 💡 **Tip:** Click **START** above and allow camera access. Perform any of the 3 signs (`hello`, `iloveyou`, `thankyou`) in front of your camera.
        """
    )

# ----------------- Mode 2: Browser Camera Snap -----------------
elif mode == "📸 Browser Camera Snap":
    st.markdown("#### Browser Webcam Snapshot")
    st.caption("Takes a photo using your web browser's camera and performs DETR sign detection.")

    snap_input = st.camera_input("Take a photo of your hand sign", key="snap_camera")

    if snap_input is not None:
        file_bytes = np.frombuffer(snap_input.getvalue(), dtype=np.uint8)
        frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame_bgr is not None:
            if mirror_input:
                frame_bgr = cv2.flip(frame_bgr, 1)

            annotated, detections, top_pred = run_detection(
                frame_bgr, model, device, transform, threshold=confidence_threshold
            )

            col1, col2 = st.columns([3, 2])
            with col1:
                st.markdown("##### Annotated Frame")
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width="stretch")

            with col2:
                st.markdown("##### Detection Breakdown")
                if detections:
                    for d in detections:
                        st.success(
                            f"### 🎯 `{d['class'].upper()}`\n\n"
                            f"**Confidence:** `{d['confidence'] * 100:.2f}%`\n\n"
                            f"**Bounding Box:** `{d['bbox']}`"
                        )
                else:
                    st.warning(
                        f"No sign detected above threshold ({int(confidence_threshold*100)}%).\n\n"
                        f"**Top prediction seen by model:** `{top_pred['class'].upper()}` with **{top_pred['confidence']*100:.1f}%** confidence.\n\n"
                        "💡 **Tip:** Lower the Confidence Threshold slider in the sidebar if your hand is far from the camera or in dim lighting."
                    )

# ----------------- Mode 3: Upload Image -----------------
elif mode == "📁 Upload Image":
    st.markdown("#### Upload an Image File")
    st.caption("Test the model against sample sign images from your computer or dataset.")

    uploaded = st.file_uploader("Upload an image (JPG / PNG)", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        img_pil = Image.open(uploaded).convert("RGB")
        frame_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        if mirror_input:
            frame_bgr = cv2.flip(frame_bgr, 1)

        annotated, detections, top_pred = run_detection(
            frame_bgr, model, device, transform, threshold=confidence_threshold
        )

        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("##### Annotated Detection")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width="stretch")

        with col2:
            st.markdown("##### Results")
            if detections:
                for idx, d in enumerate(detections, 1):
                    st.success(
                        f"**Sign #{idx}:** `{d['class'].upper()}`\n\n"
                        f"**Confidence:** `{d['confidence'] * 100:.2f}%`\n\n"
                        f"**Box:** `{d['bbox']}`"
                    )
            else:
                st.warning(
                    f"No detection above threshold ({int(confidence_threshold*100)}%).\n\n"
                    f"**Top candidate:** `{top_pred['class'].upper()}` with **{top_pred['confidence']*100:.1f}%** confidence."
                )
