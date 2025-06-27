import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Load Keras model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_model.keras")
    return model

model = load_model()

# Class names based on your class_indices: {'Alive': 0, 'Dead': 1}
class_names = ['Alive', 'Dead']

# Streamlit app
st.set_page_config(page_title="Roundworm Vitality Classifier")
st.title("🧬 Roundworm Image Classifier")
st.markdown("Upload an image of a roundworm to classify it as **Alive** or **Dead**.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file is not None:
    # Convert image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    image_resized = image.resize((128, 128))
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_label = int(prediction[0][0] > 0.5)
    predicted_class = class_names[predicted_label]
    confidence = prediction[0][0] if predicted_label == 1 else 1 - prediction[0][0]

    # Display
    st.markdown(f"### 🧠 Prediction: **{predicted_class}**")
    st.markdown(f"Confidence: `{confidence:.2f}`")

    # Model info
    st.info("Model: CNN with Data Augmentation and Class Balancing (best_model.keras)")

else:
    st.warning("Please upload an image to proceed.")

