
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from ultralytics import YOLO
import av
import cv2
import numpy as np
import threading

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(page_title="Smart Object Detection", page_icon="🎯", layout="wide")

st.title("🎯 Smart Live Object Detection & Tracking")
st.write("YOLOv8 powered real-time detection with enhanced accuracy.")

# ─── Sidebar Controls ──────────────────────────────────────────
st.sidebar.header("⚙️ Model Settings")

model_choice = st.sidebar.selectbox(
    "YOLOv8 Model (larger = more accurate)",
    ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"],
    index=1  # default to Small
)

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)
iou_threshold  = st.sidebar.slider("IOU Threshold", 0.1, 1.0, 0.5, 0.05)
imgsz          = st.sidebar.selectbox("Input Resolution", [320, 640, 1280], index=1)

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Preprocessing")
enable_denoise  = st.sidebar.checkbox("Denoise (low-light boost)", value=False)
enable_sharpen  = st.sidebar.checkbox("Sharpen Frame", value=True)
enable_contrast = st.sidebar.checkbox("Enhance Contrast (CLAHE)", value=True)

st.sidebar.markdown("---")
st.sidebar.header("📊 Live Object Count")
count_placeholder = st.sidebar.empty()
alert_placeholder  = st.sidebar.empty()

# ─── Load Model ────────────────────────────────────────────────
@st.cache_resource
def load_model(name):
    return YOLO(name)

model = load_model(model_choice)

# ─── Shared State ──────────────────────────────────────────────
lock = threading.Lock()
shared_counts = {}
alert_objects = {"cell phone", "knife", "scissors", "gun"}

# ─── Preprocessing ─────────────────────────────────────────────
def preprocess(img):
    if enable_denoise:
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    if enable_sharpen:
        kernel = np.array([[0, -1, 0],
                           [-1,  5, -1],
                           [0, -1, 0]])
        img = cv2.filter2D(img, -1, kernel)

    if enable_contrast:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = cv2.merge((clahe.apply(l), a, b))
        img = cv2.cvtColor(l, cv2.COLOR_LAB2BGR)

    return img

# ─── Video Frame Callback ───────────────────────────────────────
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Preprocess
    img = preprocess(img)

    # Run YOLOv8 tracking
    results = model.track(
        img,
        persist=True,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
        verbose=False
    )

    # Count objects
    counts = {}
    if results[0].boxes is not None:
        for box in results[0].boxes:
            label = model.names[int(box.cls)]
            counts[label] = counts.get(label, 0) + 1

    with lock:
        shared_counts.clear()
        shared_counts.update(counts)

    annotated = results[0].plot()
    return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# ─── WebRTC Streamer ────────────────────────────────────────────
ctx = webrtc_streamer(
    key="smart-detection",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)

# ─── Live Count + Alerts ────────────────────────────────────────
if ctx.state.playing:
    import time
    while ctx.state.playing:
        with lock:
            counts = dict(shared_counts)

        if counts:
            total = sum(counts.values())
            count_md = f"**Total objects: {total}**\n\n"
            count_md += "\n".join([f"- **{k}**: {v}" for k, v in sorted(counts.items())])
            count_placeholder.markdown(count_md)
        else:
            count_placeholder.info("No objects detected yet...")

        detected_alerts = alert_objects & set(counts.keys())
        if detected_alerts:
            alert_placeholder.warning("\n".join([f"⚠️ {o.title()} detected!" for o in detected_alerts]))
        else:
            alert_placeholder.empty()

        time.sleep(0.5)

# ─── Tips ───────────────────────────────────────────────────────
st.markdown("---")
with st.expander("💡 Tips for Better Detection"):
    st.markdown("""
    - **Good lighting** is the #1 factor — face a light source
    - Use **yolov8s** or **yolov8m** for better accuracy
    - **Lower confidence** (0.3–0.4) detects more objects
    - Enable **Sharpen** and **CLAHE** for clearer frames
    - Hold objects **steady** for a moment to improve tracking
    """)

st.caption("Built with YOLOv8 + Streamlit WebRTC | Activity 3 — Real-Time Object Detection")
