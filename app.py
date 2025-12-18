import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# Import semantic mappings
from semantic_mapping import SEMANTIC_MAP, GESTURE_MEANINGS

# Load trained ASL model (A–Z only)
model = load_model("sign_model.h5")

# Class labels for ASL
LABELS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

st.set_page_config(page_title="Sign language and Hand Gesture to English Translator", layout="wide")
st.title(" Sign Language to English and Gesture to English Translator")
st.markdown("---")




# =====================================================
# SECTION 1: SIGN LANGUAGE (A–Z)
# =====================================================
st.header(" Sign Language to English (A–Z)")
st.markdown("---")

sign_img = st.file_uploader(
    "Upload Sign Language Image (ASL Alphabet)",
    type=["jpg", "png", "jpeg"],
    key="sign_lang"
)

if sign_img:
    img = Image.open(sign_img).convert("L")
    st.image(img, caption="Uploaded Sign Language Image", width=250)

    if st.button("Translate Sign Language"):
        img = img.resize((28, 28))
        img_arr = np.array(img) / 255.0
        img_arr = img_arr.reshape(1, 28, 28, 1)

        pred = model.predict(img_arr)
        class_idx = np.argmax(pred)
        predicted_letter = LABELS[class_idx]

        st.success(f"Detected Sign: {predicted_letter}")

        #  SHOW ALL MEANINGS
        meanings = SEMANTIC_MAP.get(predicted_letter, [])

        st.subheader("Possible English Interpretations:")
        for meaning in meanings:
            st.write("•", meaning)

# =====================================================
# SECTION 2: GENERAL HAND GESTURES
# =====================================================
st.header(" General Hand Gestures to Meaning")
st.markdown("---")

gesture_img = st.file_uploader(
    "Upload Hand Gesture Image (Thumbs up, OK, Peace, etc.)",
    type=["jpg", "png", "jpeg"],
    key="gesture"
)

if gesture_img:
    img = Image.open(gesture_img)
    st.image(img, caption="Uploaded Hand Gesture Image", width=250)

    # Manual gesture selection (since CNN is NOT trained for gestures)
    gesture_key = st.selectbox(
        "Select the detected gesture (for now):",
        list(GESTURE_MEANINGS.keys())
    )

    if st.button("Translate Hand Gesture"):
        meaning = GESTURE_MEANINGS.get(gesture_key, "Meaning not available")
        st.subheader("Gesture Meaning:")
        st.success(meaning)

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
