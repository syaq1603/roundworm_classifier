import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

@st.cache_resource
def load_best_model():
    return load_model("best_model.keras")

model = load_best_model()

st.title("Roundworm Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).resize((128, 128))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = "Roundworm Detected" if prediction[0][0] > 0.5 else "No Roundworm Detected"
    st.write(f"**Prediction:** {result}")



