import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

st.title("🐛 Roundworm Image Classifier")

# Confirm model file exists
if not os.path.exists("best_model.keras"):
    st.error("❌ Model file 'best_model.keras' not found. Please check deployment.")
else:
    @st.cache_resource
    def load_best_model():
        return load_model("best_model.keras")

    model = load_best_model()
    st.success("✅ Model loaded successfully!")

    uploaded_file = st.file_uploader("Upload a worm image (JPEG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).resize((128, 128))
        st.image(img, caption="Uploaded Image", use_column_width=True)

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        score = prediction[0][0]
        label = "🦠 Roundworm Detected" if score > 0.5 else "✅ No Roundworm Detected"
        st.write(f"**Prediction:** {label} (Confidence: {score:.2f})")
    else:
        st.info("👈 Please upload an image to get a prediction.")
