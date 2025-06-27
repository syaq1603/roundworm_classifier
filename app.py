import streamlit as st
import numpy as np
import cv2
from keras.models import load_model

# Set page config
st.set_page_config(page_title="Roundworm Classifier", layout="centered")

st.title("🧬 Roundworm Image Classifier")
st.write("Upload a microscope image (.tif) to classify the roundworm as **Alive** or **Dead**.")

# File uploader
uploaded_file = st.file_uploader("Choose a .tif image", type=["tif"])

# Load model once
@st.cache_resource
def load_model_once():
    model = load_model("best_model.h5")
    return model

model = load_model_once()
image_size = (128, 128)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if image is None:
        st.error("Could not read the image file.")
    else:
        image_resized = cv2.resize(image, image_size) / 255.0
        image_reshaped = np.expand_dims(image_resized, axis=(0, -1))
        image_rgb = np.repeat(image_reshaped, 3, axis=-1)

        prediction = model.predict(image_rgb)
        label = "Alive" if prediction.argmax() == 0 else "Dead"
        confidence = np.max(prediction) * 100

        st.image(image, caption="Uploaded Image", use_column_width=True, channels="GRAY")
        st.success(f"**Prediction:** {label} ({confidence:.2f}% confidence)")
# Add Streamlit app for roundworm classification

