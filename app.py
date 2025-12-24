import streamlit as st
import os
import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from transformers import AutoImageProcessor, AutoModelForImageClassification

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
    - 🙂 Face Detection (PyTorch)  
    - 🧠 Deepfake Detection (HF Model)
    """)
    st.markdown("---")
    st.markdown("**Face Detector:** facenet-pytorch")
    st.markdown("**Deepfake Model:** Hugging Face")
    st.markdown("**Inference:** CPU")

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
# VIDEO UPLOAD
# =========================
st.subheader("📤 Upload Video for Verification")
uploaded_video = st.file_uploader(
    "Supported formats: MP4, AVI, MOV",
    type=["mp4", "avi", "mov"]
)

# =========================
# FRAME EXTRACTION (ADAPTIVE)
# =========================
def extract_frames(video_path, output_folder, max_cap=400):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    desired_frames = min(int(duration * 0.5), max_cap)
    interval = max(total_frames // max(desired_frames, 1), 1)

    count = saved = 0
    os.makedirs(output_folder, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            cv2.imwrite(
                os.path.join(output_folder, f"frame_{saved}.jpg"),
                frame
            )
            saved += 1
            if saved >= desired_frames:
                break
        count += 1

    cap.release()
    return saved

# =========================
# FACE EXTRACTION (FACENET-PYTORCH)
# =========================
def extract_faces(frames_folder, faces_folder):
    detector = MTCNN(keep_all=True, device="cpu")
    os.makedirs(faces_folder, exist_ok=True)
    face_count = 0

    for img_name in os.listdir(frames_folder):
        img_path = os.path.join(frames_folder, img_name)
        image = Image.open(img_path).convert("RGB")

        boxes, _ = detector.detect(image)
        if boxes is None:
            continue

        img_np = np.array(image)
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = img_np[y1:y2, x1:x2]
            if face.size == 0:
                continue

            Image.fromarray(face).save(
                os.path.join(faces_folder, f"face_{face_count}.jpg")
            )
            face_count += 1

    return face_count

# =========================
# DEEPFAKE DETECTOR (HUGGING FACE)
# =========================
class DeepfakeDetector:
    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained(
            "dima806/deepfake_detection"
        )
        self.model = AutoModelForImageClassification.from_pretrained(
            "dima806/deepfake_detection"
        )
        self.model.eval()

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

        return probs[0][0].item(), probs[0][1].item()

# =========================
# AGGREGATION
# =========================
def analyze_faces(faces_folder):
    detector = DeepfakeDetector()
    real_scores, fake_scores = [], []

    for face in os.listdir(faces_folder):
        real, fake = detector.predict(
            os.path.join(faces_folder, face)
        )
        real_scores.append(real)
        fake_scores.append(fake)

    return (
        float(np.mean(real_scores)),
        float(np.mean(fake_scores)),
        float(np.var(fake_scores))
    )

# =========================
# PIPELINE EXECUTION
# =========================
if uploaded_video:
    progress = st.progress(0)

    video_path = os.path.join(UPLOAD_FOLDER, uploaded_video.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())

    st.video(video_path)
    progress.progress(20)

    name = os.path.splitext(uploaded_video.name)[0]
    frames_path = os.path.join(FRAMES_FOLDER, name)
    faces_path = os.path.join(FACES_FOLDER, name)

    frames = extract_frames(video_path, frames_path)
    progress.progress(40)

    faces = extract_faces(frames_path, faces_path)
    progress.progress(60)

    st.subheader("🧠 Deepfake Analysis")
    real, fake, var = analyze_faces(faces_path)
    progress.progress(100)

    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 Real Probability", f"{real:.2f}")
    c2.metric("🔴 Fake Probability", f"{fake:.2f}")
    c3.metric("📊 Consistency", f"{var:.4f}")

    st.markdown("### 🧾 Final Verdict")
    margin = abs(fake - real)

    if fake > real and margin > 0.1:
        st.error("🚨 High Confidence Deepfake Detected")
    elif real > fake and margin > 0.1:
        st.success("✅ Media Appears Authentic")
    else:
        st.warning("⚠️ Uncertain – Manual Review Recommended")

    st.markdown("### 🧠 Explanation")
    st.write(
        f"""
        The system analyzed **{faces} face samples** sampled across the entire video.
        The decision is based on **aggregated predictions** and **temporal consistency**.
        """
    )

    with st.expander("👀 Preview Extracted Faces"):
        for face in os.listdir(faces_path)[:5]:
            st.image(
                os.path.join(faces_path, face),
                width=160
            )

    st.caption("© TrustNet – Hackathon Prototype")





