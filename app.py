import os, sys, subprocess

# Force install headless OpenCV (in case ultralytics pulled GUI version)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python"], stdout=subprocess.DEVNULL)
subprocess.run([sys.executable, "-m", "pip", "install", "-U", "opencv-python-headless==4.10.0.84"], stdout=subprocess.DEVNULL)

# Prevent GUI dependencies (no display, no OpenGL)
os.environ["DISPLAY"] = ""
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "0"
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO
import av
import cv2
import os

st.set_page_config(page_title="ASL Detection", layout="wide")

st.title("🤟 Real-Time American Sign Language Detection")
st.markdown(
    """
    This Streamlit app uses your webcam and a fine-tuned **YOLOv11** model 
    to detect American Sign Language (ASL) gestures in real-time.
    """
)

# ✅ Load YOLOv11 model safely
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "best.pt")
    model = YOLO(model_path)
    return model

model = load_model()
st.success("✅ YOLOv11 model loaded successfully!")

# ✅ Video processor for live ASL detection
class ASLProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model

    def recv(self, frame):
        # Convert incoming video frame to ndarray
        img = frame.to_ndarray(format="bgr24")

        # Run YOLOv11 detection (lower conf = more sensitivity)
        results = self.model.predict(img, conf=0.25, verbose=False)

        # Draw bounding boxes + labels
        annotated = results[0].plot(labels=True)

        # Optional: Print detections in app sidebar (debugging)
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            detected_labels = [self.model.names[int(box.cls)] for box in boxes]
            st.sidebar.write("🖐 Detected:", detected_labels)

        # Return processed frame
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# ✅ Start webcam stream (optimized for phones)
webrtc_streamer(
    key="asl-stream",
    video_processor_factory=ASLProcessor,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 1280},
            "height": {"ideal": 720}
        },
        "audio": False
    },
)

st.markdown(
    """
    ---
    🧩 **Instructions:**
    1. Tap **Start** and allow camera access on your phone.  
    2. Hold your hand sign close to the camera in good lighting.  
    3. Watch the bounding boxes and detected labels in real time.  
    4. Detected sign names will appear in the sidebar.  
    """
)


