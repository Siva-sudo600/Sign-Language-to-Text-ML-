import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# ✅ IMPORT SEMANTIC MAP
from semantic_mapping import SEMANTIC_MAP

# Load trained model
model = load_model("sign_model.h5")

alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

st.title("Sign Language to English Translator")

uploaded = st.file_uploader(
    "Upload Hand Sign Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded is not None:
    image = Image.open(uploaded).convert("L")
    st.image(image, caption="Uploaded Image", width=300)

    img = np.array(image)
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    if st.button("Translate"):
        prediction = model.predict(img)
        index = np.argmax(prediction)
        letter = alphabet[index]

        st.success(f"Detected Sign: {letter}")
        st.subheader("English Meanings:")

        for meaning in SEMANTIC_MAP.get(letter, ["No meaning found"]):
            st.write("•", meaning)

