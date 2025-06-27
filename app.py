import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
import numpy as np
import io

# Load the best model
model = tf.keras.models.load_model("best_model.keras")

# Streamlit app
st.title("Roundworm Classifier 🐛")
st.write("Upload an image of a roundworm and the model will classify it.")

# Upload image
uploaded_file = st.file_uploader(
    "Upload a worm image (JPG, PNG, or TIF)", 
    type=["jpg", "jpeg", "png", "tif", "tiff"]
)

if uploaded_file is not None:
    try:
        # Read image
        image = Image.open(uploaded_file)

        # Convert TIF or other modes to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        image = image.resize((128, 128))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Predict
        prediction = model.predict(image_array)
        predicted_class = "Infected" if prediction[0][0] > 0.5 else "Healthy"

        st.subheader(f"🧠 Prediction: **{predicted_class}** ({prediction[0][0]:.2f} probability)")

    except UnidentifiedImageError:
        st.error("❌ The uploaded file is not a valid image.")

