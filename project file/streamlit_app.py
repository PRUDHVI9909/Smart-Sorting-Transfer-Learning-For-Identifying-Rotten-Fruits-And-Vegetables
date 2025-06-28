import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Custom CSS for styling
st.markdown("""
    <style>
    .main-title {
        color: #2E8B57;
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .result-text {
        color: #1E90FF;
        font-size: 1.5em;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
    }
    .stButton>button {
        background-color: #2E8B57;
        color: white;
        font-size: 1.1em;
        border-radius: 8px;
    }
    .stFileUploader label {
        color: #2E8B57;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Class labels (update if needed)
class_labels = [
    'Apple Healthy', 'Apple Rotten', 'Banana Healthy', 'Banana Rotten',
    'Carrot Healthy', 'Carrot Rotten', 'Potato Healthy', 'Potato Rotten'
]

# Load model
@st.cache_resource
def load_saved_model():
    return load_model('fruit_vegetable_disease_model.keras')

model = load_saved_model()

# Title
st.markdown('<div class="main-title">Fruit & Vegetable Disease Classifier</div>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload an image of a fruit or vegetable", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image as bytes and convert to numpy array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_resized = cv2.resize(img, (128, 128))
    img_input = img_resized.astype('float32') / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    # Display image
    st.image(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)

    # Add Predict button
    if st.button("Predict"):
        prediction = model.predict(img_input)
        predicted_class = class_labels[np.argmax(prediction)]
        # Display result
        st.markdown(f'<div class="result-text">Prediction: {predicted_class}</div>', unsafe_allow_html=True)
