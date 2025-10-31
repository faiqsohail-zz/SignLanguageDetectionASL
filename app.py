import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import time

st.set_page_config(page_title="ASL Detection", layout="wide")

# Load model once
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # your fine-tuned model
    return model

model = load_model()

st.title("🤟 Real-time American Sign Language Detection")
st.write("This app uses your webcam and YOLOv11 model to detect ASL gestures in real-time.")

# Start webcam
run = st.checkbox('Start Webcam')
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.warning("Could not access webcam.")
        break

    frame = cv2.flip(frame, 1)  # mirror view for natural feeling

    # Inference
    results = model.predict(source=frame, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()

    FRAME_WINDOW.image(annotated_frame, channels="BGR")

camera.release()
st.write("✅ Detection stopped.")
