import streamlit as st
import os
import numpy as np
import torch
from PIL import Image
from mtcnn import MTCNN
from transformers import AutoImageProcessor, AutoModelForImageClassification
import imageio

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="TrustNet – Deepfake Detection",
    page_icon="🛡️",
    layout="wide"
)

# =========================
# HEADER
# =========================
st.markdown(
    """
    <h1 style='text-align: center;'>🛡️ TrustNet</h1>
    <h4 style='text-align: center; color: gray;'>
    Agentic AI for Deepfake Detection & Authenticity Verification
    </h4>
    <hr>
    """,
    unsafe_allow_html=True
)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("⚙️ System Overview")
    st.markdown("""
    **Pipeline**
    - 🎥 Video Upload  
    - 🖼️ Adaptive Frame Sampling  
    - 🙂 Face Detection  
    - 🧠 Deepfake Analysis  
    - 🧾 Explainable Verdict  
    """)
    st.markdown("---")
    st.markdown("**CV Backend**: Pure Python (Cloud-safe)")
    st.success("✅ Streamlit Cloud Compatible")

# =========================
# FOLDERS
# =========================
UPLOAD_FOLDER = "uploaded_videos"
FRAMES_FOLDER = "extracted_frames"
FACES_FOLDER = "extracted_faces"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)
os.makedirs(FACES_FOLDER, exist_ok=True)

# =========================
# UPLOAD
# =========================
st.subheader("📤 Upload Media for Verification")
uploaded_video = st.file_uploader(
    "Supported formats: MP4, AVI, MOV",
    type=["mp4", "avi", "mov"]
)

# =========================
# FRAME EXTRACTION (NO OPENCV)
# =========================
def extract_frames(video_path, output_folder, max_cap=400):
    os.makedirs(output_folder, exist_ok=True)

    reader = imageio.get_reader(video_path)
    meta = reader.get_meta_data()

    fps = meta.get("fps", 30)
    total_frames = meta.get("nframes", 300)

    duration_sec = total_frames / fps
    desired_frames = min(int(duration_sec * 0.5), max_cap)
    interval = max(total_frames // max(desired_frames, 1), 1)

    saved = 0
    for i, frame in enumerate(reader):
        if i % interval == 0:
            Image.fromarray(frame).save(
                os.path.join(output_folder, f"frame_{saved}.jpg")
            )
            saved += 1
            if saved >= desired_frames:
                break

    reader.close()
    return saved

# =========================
# FACE EXTRACTION (MTCNN)
# =========================
def extract_faces(frames_folder, faces_folder):
    detector = MTCNN()
    os.makedirs(faces_folder, exist_ok=True)
    face_count = 0

    for img_name in os.listdir(frames_folder):
        img_path = os.path.join(frames_folder, img_name)
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)

        faces = detector.detect_faces(image_np)
        for face in faces:
            x, y, w, h = face["box"]
            x, y = max(0, x), max(0, y)

            face_img = image_np[y:y+h, x:x+w]
            if face_img.size == 0:
                continue

            Image.fromarray(face_img).save(
                os.path.join(faces_folder, f"face_{face_count}.jpg")
            )
            face_count += 1

    return face_count

# =========================
# BLUR / TEXTURE ANALYSIS (NO OPENCV)
# =========================
def blur_score(image_path):
    img = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
    gy, gx = np.gradient(img)
    return np.mean(gx**2 + gy**2)

# =========================
# DEEPFAKE MODEL
# =========================
@st.cache_resource
def load_model():
    model_name = "prithivMLmods/Deep-Fake-Detector-Model"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model.eval()
    return processor, model

class DeepfakeDetector:
    def __init__(self):
        self.processor, self.model = load_model()

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]

        fake_prob = probs[0].item()
        real_prob = probs[1].item()
        return real_prob, fake_prob

# =========================
# AGGREGATION
# =========================
def analyze_faces(faces_folder):
    detector = DeepfakeDetector()
    real_scores, fake_scores, blur_scores = [], [], []

    face_files = os.listdir(faces_folder)
    if not face_files:
        return None, None, None, None

    for face in face_files:
        face_path = os.path.join(faces_folder, face)
        real, fake = detector.predict(face_path)
        real_scores.append(real)
        fake_scores.append(fake)
        blur_scores.append(blur_score(face_path))

    return (
        float(np.mean(real_scores)),
        float(np.mean(fake_scores)),
        float(np.var(fake_scores)),
        float(np.mean(blur_scores))
    )

# =========================
# PIPELINE
# =========================
if uploaded_video is not None:
    st.markdown("### 🔄 Processing Pipeline")
    progress = st.progress(0)

    video_path = os.path.join(UPLOAD_FOLDER, uploaded_video.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())

    st.video(video_path)
    progress.progress(20)

    video_name = os.path.splitext(uploaded_video.name)[0]
    frames_output = os.path.join(FRAMES_FOLDER, video_name)
    faces_output = os.path.join(FACES_FOLDER, video_name)

    extract_frames(video_path, frames_output)
    progress.progress(40)

    faces_count = extract_faces(frames_output, faces_output)
    progress.progress(60)

    st.subheader("🧠 Deepfake Analysis")
    real_score, fake_score, variance, avg_blur = analyze_faces(faces_output)
    progress.progress(100)

    if real_score is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("🟢 Real Probability", f"{real_score:.2f}")
        col2.metric("🔴 Fake Probability", f"{fake_score:.2f}")
        col3.metric("📊 Consistency", f"{variance:.4f}")

        st.markdown("### 🧾 Final Verdict")
        margin = abs(fake_score - real_score)

        if fake_score > real_score and margin > 0.1:
            st.error("🚨 High Confidence Deepfake Detected")
        elif real_score > fake_score and margin > 0.1:
            st.success("✅ Media Appears Authentic")
        else:
            st.warning("⚠️ Uncertain — Manual Review Recommended")

        st.markdown("### 🧠 Why this verdict?")
        explanations = [
            f"Analyzed {faces_count} face samples across the video.",
            "Predictions were consistent across frames." if variance < 0.02 else
            "Predictions varied across frames.",
            "Detected abnormal smoothness in facial textures." if avg_blur < 100 else
            "Facial texture sharpness appears natural.",
            "Model confidence separation was strong." if margin > 0.3 else
            "Model confidence separation was marginal."
        ]

        for exp in explanations:
            st.write("•", exp)

        with st.expander("👀 Preview Extracted Faces"):
            face_files = os.listdir(faces_output)[:5]
            cols = st.columns(len(face_files))
            for col, face in zip(cols, face_files):
                col.image(os.path.join(faces_output, face), width=150)

    st.markdown("---")
    st.caption("© TrustNet – Hackathon Prototype | Cloud-Safe Explainable AI")



