import streamlit as st
import pandas as pd
import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image
import os
import tempfile

from streamlit_webrtc import (
    webrtc_streamer,
    VideoTransformerBase,
    WebRtcMode,
    RTCConfiguration,
)

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
# RTC CONFIG (STUN + TURN)
# =========================
def get_rtc_configuration():
    # STUN: free direct connect
    stun_servers = [
        "stun:stun.l.google.com:19302",
        "stun:stun1.l.google.com:19302",
        "stun:stun2.l.google.com:19302",
        "stun:stun3.l.google.com:19302",
        "stun:stun4.l.google.com:19302",
    ]

    ice_servers = [{"urls": stun_servers}]

    # TURN: relay (fixes strict WiFi / college networks)
    # NOTE: TURN creds must be set in Streamlit secrets
    try:
        turn_url = st.secrets["TURN_URL"]
        turn_username = st.secrets["TURN_USERNAME"]
        turn_password = st.secrets["TURN_PASSWORD"]

        ice_servers.append(
            {
                "urls": [turn_url],
                "username": turn_username,
                "credential": turn_password,
            }
        )
        st.sidebar.success("✅ TURN enabled (best connectivity)")
    except Exception:
        st.sidebar.warning("⚠ TURN not configured, using only STUN")

    return RTCConfiguration({"iceServers": ice_servers})


RTC_CONFIGURATION = get_rtc_configuration()


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


# ============================================================
# WEBCAM MODE (WebRTC)
# ============================================================
class EmotionVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0
        self.last_faces = 0
        self.last_emotion = "None"
        self.last_conf = 0.0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        self.last_faces = len(faces)

        for (x, y, w, h) in faces:
            face = img[y:y + h, x:x + w]
            emotion, conf = predict_face_emotion(face)

            self.last_emotion = emotion
            self.last_conf = conf

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{emotion} ({conf:.2f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        return img


# =========================
# WEBCAM MODE
# =========================
if mode == "Webcam (Face Only)":

    st.warning("🎥 Webcam mode analyzes **facial emotion only** (no audio).")
    st.info("✅ Allow camera permission in browser. TURN is needed if your WiFi blocks WebRTC.")

    stats_placeholder = st.empty()

    ctx = webrtc_streamer(
        key="emotion-webcam",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=EmotionVideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration=RTC_CONFIGURATION,
    )

    if ctx.video_transformer:
        stats_placeholder.markdown(
            f"""
            **Frames Processed:** {ctx.video_transformer.frame_count}  
            **Faces Detected:** {ctx.video_transformer.last_faces}  
            **Last Emotion:** {ctx.video_transformer.last_emotion}  
            **Confidence:** {ctx.video_transformer.last_conf:.2f}
            """
        )


# =========================
# VIDEO UPLOAD MODE
# =========================
if mode == "Upload Video (Face + Voice)":

    video_file = st.file_uploader(
        "📤 Upload a video file",
        type=["mp4", "avi", "mov"]
    )

    if video_file:
        # ✅ show preview on Streamlit
        st.video(video_file)

        # ✅ safe temp path for Streamlit Cloud
        temp_dir = tempfile.gettempdir()
        temp_video_path = os.path.join(temp_dir, "temp_video.mp4")

        with open(temp_video_path, "wb") as f:
            f.write(video_file.read())

        cap = cv2.VideoCapture(temp_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # ✅ handle codec problems (cloud)
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
