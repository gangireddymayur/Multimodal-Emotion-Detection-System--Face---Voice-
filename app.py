import streamlit as st
import pandas as pd
import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image
import os
import tempfile

from model import build_model
from audio_utils import extract_audio
from voice_emotion import predict_voice_emotion
from fusion import fuse_emotions

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Multimodal Emotion Detection",
    page_icon="🎭",
    layout="wide"
)

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMOTIONS = [
    "Surprise", "Fear", "Disgust",
    "Happy", "Sad", "Angry", "Neutral"
]

# =========================
# CLOUD DETECTION (Only for webcam fix)
# =========================
def is_streamlit_cloud():
    # Streamlit Cloud sets some environment variables; this is safe detection
    return (
        os.environ.get("STREAMLIT_SHARING") == "true"
        or "streamlit" in os.environ.get("HOSTNAME", "").lower()
        or os.environ.get("HOME", "") == "/home/adminuser"
    )

IS_CLOUD = is_streamlit_cloud()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🎭 Emotion AI Dashboard")
st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "Select Input Mode",
    ["Webcam (Face Only)", "Upload Video (Face + Voice)"]
)

# =========================
# LOAD FACE MODEL
# =========================
@st.cache_resource
def load_face_model():
    model = build_model(num_classes=7)
    model.load_state_dict(
        torch.load("best_efficientnet.pth", map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()
    return model

face_model = load_face_model()

# =========================
# TRANSFORMS
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# FACE DETECTOR
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =========================
# FACE EMOTION PREDICTION
# =========================
def predict_face_emotion(face_img):
    img = Image.fromarray(face_img).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = face_model(x)
        probs = torch.softmax(logits, dim=1)
        idx = torch.argmax(probs, dim=1).item()

    return EMOTIONS[idx], probs[0][idx].item()

# =========================
# HEADER
# =========================
st.title("🎭 Multimodal Emotion Detection System")
st.caption("Facial Emotion Recognition + Voice Emotion Fusion")

st.markdown("---")

# =========================
# WEBCAM MODE
# =========================
if mode == "Webcam (Face Only)":

    st.warning("🎥 Webcam mode analyzes **facial emotion only** (no audio).")

    # ✅ FIX: Webcam will never work on Streamlit Cloud server (no webcam device)
    if IS_CLOUD:
        st.error(
            "❌ Webcam mode cannot work on Streamlit Cloud because the server has no webcam device.\n\n"
            "✅ Run this project locally (PyCharm / terminal) to use webcam mode."
        )
        st.stop()

    run = st.checkbox("▶ Start Webcam")
    frame_placeholder = st.empty()
    stats_placeholder = st.empty()

    cap = cv2.VideoCapture(0)
    frame_count = 0

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Cannot access webcam. Please check camera permissions / device.")
            break

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            emotion, conf = predict_face_emotion(face)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{emotion} ({conf:.2f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        frame_placeholder.image(frame, channels="BGR")

        stats_placeholder.markdown(
            f"""
            **Frames Processed:** {frame_count}  
            **Faces Detected:** {len(faces)}
            """
        )

    cap.release()

# =========================
# VIDEO UPLOAD MODE
# =========================
if mode == "Upload Video (Face + Voice)":

    video_file = st.file_uploader(
        "📤 Upload a video file",
        type=["mp4", "avi", "mov"]
    )

    if video_file:
        # ✅ FIX: show video preview (works on Streamlit Cloud)
        st.video(video_file)

        # ✅ FIX: Streamlit Cloud needs safe temp path
        temp_dir = tempfile.gettempdir()
        temp_video_path = os.path.join(temp_dir, "temp_video.mp4")

        with open(temp_video_path, "wb") as f:
            f.write(video_file.read())

        cap = cv2.VideoCapture(temp_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # ✅ FIX: handle broken fps/frames on cloud codecs
        if fps is None or fps == 0:
            fps = 25.0
        if total_frames is None or total_frames <= 0:
            total_frames = 1

        st.success(f"📹 Video Loaded | FPS: {fps:.1f} | Frames: {total_frames}")

        frame_placeholder = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()

        emotion_counts = {e: 0 for e in EMOTIONS}
        confidence_sum = {e: 0.0 for e in EMOTIONS}

        frame_idx = 0

        # =========================
        # FACE PROCESSING LOOP
        # =========================
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            progress_bar.progress(min(int((frame_idx / total_frames) * 100), 100))
            status_text.text(f"Processing frame {frame_idx}/{total_frames}")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face = frame[y:y + h, x:x + w]
                emotion, conf = predict_face_emotion(face)

                emotion_counts[emotion] += 1
                confidence_sum[emotion] += conf

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    f"{emotion} ({conf:.2f})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 0, 0),
                    2
                )

            frame_placeholder.image(frame, channels="BGR")

        cap.release()
        progress_bar.empty()
        status_text.empty()

        # =========================
        # FACE RESULTS
        # =========================
        total_detections = sum(emotion_counts.values())

        if total_detections == 0:
            st.warning("⚠ No faces detected in this video.")
        else:
            dominant_emotion = max(emotion_counts, key=emotion_counts.get)
            avg_conf = confidence_sum[dominant_emotion] / max(
                emotion_counts[dominant_emotion], 1
            )

            result_df = pd.DataFrame({
                "Emotion": emotion_counts.keys(),
                "Detections": emotion_counts.values(),
                "Avg Confidence": [
                    confidence_sum[e] / max(emotion_counts[e], 1)
                    for e in EMOTIONS
                ]
            })

            st.markdown("---")
            st.subheader("📊 Facial Emotion Summary")

            col1, col2 = st.columns(2)
            col1.metric("Dominant Emotion", dominant_emotion)
            col2.metric("Avg Confidence", f"{avg_conf:.2f}")

            st.dataframe(result_df, use_container_width=True)
            st.bar_chart(result_df.set_index("Emotion")["Detections"])

            # =========================
            # VOICE EMOTION
            # =========================
            st.info("🎧 Analyzing voice emotion...")
            audio_path = extract_audio(temp_video_path)
            voice_emotion, voice_conf = predict_voice_emotion(audio_path)

            st.subheader("🎧 Voice Emotion Result")
            st.write(f"**Emotion:** {voice_emotion}")
            st.write(f"**Confidence:** {voice_conf:.2f}")

            # =========================
            # FUSION
            # =========================
            final_emotion = fuse_emotions(
                dominant_emotion,
                avg_conf,
                voice_emotion,
                voice_conf
            )

            st.subheader("🧠 Final Multimodal Emotion")
            st.success(f"🎯 **FINAL EMOTION:** {final_emotion}")

            st.download_button(
                "⬇ Download Results (CSV)",
                result_df.to_csv(index=False),
                "emotion_results.csv"
            )
