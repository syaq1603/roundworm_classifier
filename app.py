import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Configure the page
st.set_page_config(
    page_title="Roundworm Classifier",
    page_icon="🧬",
    layout="centered"
)

# Main title and project description
st.markdown(
    "<h3 style='text-align: center; color: #4B8BBE;'>🧬 Roundworm Life Status Classifier</h3>",
    unsafe_allow_html=True
)

st.markdown("""
This Streamlit app classifies microscopic roundworm images as either **Alive** or **Dead**.  
The model was trained using a **CNN with SMOTE** to improve class balance, achieving **96% accuracy**.
""")

# Load the best model (cached to avoid reloading)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.keras")

model = load_model()

# Image upload section
st.markdown(
    "<h4 style='color: #6c757d;'>📤 Upload Roundworm Image</h4>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload a JPG, PNG, or TIFF image", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=128)

    # Preprocess
    resized_image = image.resize((128, 128))
    img_array = np.array(resized_image) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_batch)[0][0]
    label = "Alive" if prediction < 0.5 else "Dead"
    confidence = 1 - prediction if prediction < 0.5 else prediction

    # Display result
    st.header("🔍 Prediction Result")
    st.success(f"**Prediction:** {label}")
    st.write(f"**Confidence:** {confidence:.2%}")

    # Model info
    st.markdown("---")
    st.markdown("📌 **Model Used:** Convolutional Neural Network (CNN) + SMOTE")

