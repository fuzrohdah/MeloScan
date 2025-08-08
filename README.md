# OncoScanSkin (Previously MeloScan)

**OncoScanSkin** is a lightweight Streamlit app that uses a Convolutional Neural Network (CNN) to classify dermoscopic images of skin lesions as **benign** or **malignant** (melanoma). It supports real-time webcam prediction and image upload.

---

## Features

- Real-time webcam-based skin lesion classification  
- Upload image for on-demand prediction  
- Uses MobileNetV2-based CNN for efficient inference  
- Preprocessing includes resizing and MobileNet-style normalization  
- Displays prediction label (Benign/Malignant) with confidence score  
- Trained on ~10,000 high-resolution dermoscopic images

---

## Tech Stack

- Python  
- TensorFlow / Keras  
- OpenCV  
- Streamlit  
- PIL (Python Imaging Library)

---

## How to Run

1. Clone this repository.
2. Place your `skin_lesion_classifier.h5` model in the project directory.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
