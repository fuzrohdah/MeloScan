# MELOSCAN



import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load your trained model
MODEL_PATH = r"C:\Users\haris\OneDrive\Desktop\melanoma\skin_lesion_classifier.h5"  # change this if using a different path
model = load_model(MODEL_PATH)

st.title("MeloScan: Melanoma vs Similar Looking Benign Lesions Classifier")
st.markdown("Detects **benign** vs **malignant** from webcam feed or uploaded images")
st.markdown("Note: This model is trained on nearly 10000 images of melanoma and benign lesions. It is not trained for detecting other vaguely similar/dissimilar skin lesions like Tinea, bruises etc. Eg If you upload the picture of a laptop, it might classify it as MELANOMA because it doesnt know what a laptop looks like!")


option = st.radio("Choose input mode:", ["Webcam", "Upload Image"])

# Define preprocessing function
def preprocess_frame(frame):
    image = cv2.resize(frame, (224, 224))
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

# Prediction function
def predict(frame):
    preprocessed = preprocess_frame(frame)
    prob = model.predict(preprocessed)[0][0]
    label = "Malignant" if prob > 0.5 else "Benign"
    confidence = prob if prob > 0.5 else 1 - prob
    return label, confidence

# Webcam Mode
if option == "Webcam":
    run = st.checkbox("Start Camera")
    FRAME_WINDOW = st.image([])
    cap = None

    if run:
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Camera not found.")
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            label, confidence = predict(frame_rgb)
            cv2.putText(frame_rgb, f"{label} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            FRAME_WINDOW.image(frame_rgb)

        cap.release()

# Upload Mode
else:
    uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        frame = np.array(image)
        label, confidence = predict(frame)
        st.image(image, caption=f"Prediction: {label} ({confidence:.2f})", use_column_width=True)
