import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO
import av
import cv2

st.set_page_config(page_title="ASL Detection", layout="wide")

st.title("🤟 Real-Time American Sign Language Detection")
st.markdown(
    """
    This Streamlit app uses your webcam and a fine-tuned **YOLOv11** model 
    to detect American Sign Language (ASL) gestures in real-time.
    """
)

# Load YOLOv11 model (your fine-tuned best.pt)
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # Ensure best.pt is in the same folder
    return model

model = load_model()

# Video processor for live detection
class ASLProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model

    def recv(self, frame):
        # Convert video frame to array
        img = frame.to_ndarray(format="bgr24")

        # Run YOLOv11 detection
        results = self.model.predict(img, conf=0.5, verbose=False)

        # Draw bounding boxes
        annotated = results[0].plot()

        # Return processed frame to Streamlit
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# Start live video streaming from browser webcam
webrtc_streamer(
    key="asl-stream",
    video_processor_factory=ASLProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

st.markdown(
    """
    ---
    🧩 **Instructions:**
    1. Click **Start** to allow webcam access.  
    2. Show your ASL hand signs in front of the camera.  
    3. The YOLOv11 model will detect and label them in real-time.
    """
)
