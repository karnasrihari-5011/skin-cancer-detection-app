import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(page_title="Skin Cancer Detection", layout="centered")

st.title("🧬 Skin Cancer Detection using Deep Learning")

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("skin_cancer_model.keras")

model = load_model()

# ----------------------------
# Upload Image
# ----------------------------
uploaded_file = st.file_uploader("Upload Skin Lesion Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Display Uploaded Image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess Image
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ----------------------------
    # Prediction
    # ----------------------------
    prediction = model.predict(img_array)[0][0]

    # Binary classification logic
    if prediction > 0.5:
        class_label = "Melanoma (Malignant Skin Cancer)"
        confidence = prediction
    else:
        class_label = "Benign (Non-Cancerous)"
        confidence = 1 - prediction

    # ----------------------------
    # Invalid Image Handling
    # ----------------------------
    if confidence < 0.70:
        st.warning("⚠️ Unable to confidently classify. Please upload a clear skin lesion image.")
    
    else:
        st.subheader("🔎 Prediction Result")

        # Disease Name
        st.write(f"**Disease:** {class_label}")

        # Confidence Percentage
        st.write(f"**Confidence:** {round(confidence * 100, 2)}%")

        # Progress Bar
        st.progress(int(confidence * 100))

        # Red Alert only for Melanoma
        if "Melanoma" in class_label:
            st.error("🚨 RED ALERT: Possible Melanoma Detected! Please consult a dermatologist immediately.")
