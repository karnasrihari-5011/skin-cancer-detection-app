import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Must match training size
IMG_SIZE = 224  

st.set_page_config(page_title="Skin Cancer Detection", layout="centered")

st.title("🧬 Skin Cancer Detection System")
st.write("Upload a dermoscopic skin lesion image for analysis")

# Load model
model = tf.keras.models.load_model("skin_cancer_binary_model.keras")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)[0][0]

    benign_conf = (1 - prediction) * 100
    malignant_conf = prediction * 100

    st.markdown("---")
    st.subheader("🔍 Prediction Result")

    # Progress Bar
    st.write("Model Confidence Level")
    st.progress(int(max(benign_conf, malignant_conf)))

    # Output Section
    if prediction > 0.5:
        st.error("🚨 Malignant Skin Lesion Detected")
        st.write("**Full Disease Name:** Melanoma (Skin Cancer)")
        st.write(f"**Confidence:** {malignant_conf:.2f}%")

        # Red Alert only for melanoma
        st.warning("⚠️ Immediate medical consultation is strongly recommended.")
        
    else:
        st.success("✅ Benign Skin Lesion Detected")
        st.write("**Full Disease Name:** Non-Cancerous (Benign Lesion)")
        st.write(f"**Confidence:** {benign_conf:.2f}%")

st.markdown("---")
st.caption("⚠️ This AI tool is for educational purposes only and not a medical diagnosis.")