import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
from PIL import Image
import numpy as np

# -----------------------------
# Streamlit page setup
# -----------------------------
st.set_page_config(page_title="Sign Language Detection", layout="centered")
st.title("🖐️ American Sign Language Detection")
st.write("Upload an **image** or **video** to detect hand signs using your fine-tuned YOLOv11n model.")

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    model_path = "best.pt"  # ensure best.pt is in the same folder
    model = YOLO(model_path)
    return model

model = load_model()

# -----------------------------
# File uploader
# -----------------------------
uploaded_file = st.file_uploader("Upload an Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

if uploaded_file:
    file_type = uploaded_file.type

    # -----------------------------
    # Handle Image
    # -----------------------------
    if file_type.startswith("image"):
        image = Image.open(uploaded_file).convert("RGB")

        # Resize to 640x640
        image_resized = image.resize((640, 640))
        st.image(image_resized, caption="Uploaded Image (resized)", use_column_width=True)

        with st.spinner("Detecting hand signs..."):
            results = model.predict(source=np.array(image_resized), imgsz=640, conf=0.5)

        # Get annotated result
        result_img = results[0].plot()[:, :, ::-1]
        st.image(result_img, caption="Detection Result", use_column_width=True)

    # -----------------------------
    # Handle Video
    # -----------------------------
    elif file_type.startswith("video"):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        st.video(video_path)
        st.write("Processing video... This may take a while ⏳")

        output_path = "output.mp4"
        with st.spinner("Detecting hand signs..."):
            model.predict(
                source=video_path,
                save=True,
                save_txt=False,
                conf=0.5,
                imgsz=640
            )

        # Find latest output video
        from glob import glob
        output_dirs = sorted(glob("runs/detect/predict*"), key=os.path.getmtime)
        output_dir = output_dirs[-1]
        output_path = os.path.join(output_dir, os.path.basename(video_path))

        st.success("✅ Detection complete!")
        st.video(output_path)
