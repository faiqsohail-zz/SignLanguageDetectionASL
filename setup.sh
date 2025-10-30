#!/bin/bash
echo "🔧 Cleaning up OpenCV conflicts..."
pip uninstall -y opencv-python || true
pip install opencv-python-headless==4.12.0.88 --force-reinstall
echo "🚀 Launching Streamlit app..."
streamlit run app.py
