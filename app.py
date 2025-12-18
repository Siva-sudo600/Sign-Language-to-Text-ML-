# import streamlit as st
# import numpy as np
# import cv2
# from tensorflow.keras.models import load_model
# from PIL import Image

# # ✅ IMPORT SEMANTIC MAP
# from semantic_mapping import SEMANTIC_MAP

# # Load trained model
# model = load_model("model.h5")

# alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# st.title("Sign Language to English Translator")

# uploaded = st.file_uploader(
#     "Upload Hand Sign Image",
#     type=["jpg", "png", "jpeg"]
# )

# if uploaded is not None:
#     image = Image.open(uploaded).convert("L")
#     st.image(image, caption="Uploaded Image", width=300)

#     img = np.array(image)
#     img = cv2.resize(img, (28, 28))
#     img = img / 255.0
#     img = img.reshape(1, 28, 28, 1)

#     if st.button("Translate"):
#         prediction = model.predict(img)
#         index = np.argmax(prediction)
#         letter = alphabet[index]

#         st.success(f"Detected Sign: {letter}")
#         st.subheader("English Meanings:")

#         for meaning in SEMANTIC_MAP.get(letter, ["No meaning found"]):
#             st.write("•", meaning)


import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model


from semantic_mapping import SEMANTIC_MAP, GESTURE_MEANINGS

# Load trained sign model
model = load_model("sign_model.h5")
alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


st.title(" Hand Gesture & Sign Language Translator")

# ======================================
# SECTION 1: SIGN LANGUAGE (A–Z)
# ======================================
st.header("Sign Language to English Text")

sign_image = st.file_uploader(
    "Upload Sign Language Image",
    type=["jpg", "png", "jpeg"],
    key="sign"
)

if sign_image:
    image = Image.open(sign_image).convert("L")
    st.image(image, width=250)

    img = np.array(image)
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    if st.button("Translate Sign"):
        pred = model.predict(img)
        letter = alphabet[np.argmax(pred)]

        st.success(f"Detected Sign: {letter}")
        st.write("### Meaning:")
        for m in SEMANTIC_MAP.get(letter, []):
            st.write("•", m)

# ======================================
# SECTION 2: GENERAL HAND GESTURES
# ======================================
st.header(" General Hand Gestures to Text ")

gesture_image = st.file_uploader(
    "Upload Hand Gesture Image",
    type=["jpg", "png", "jpeg"],
    key="gesture"
)

def detect_gesture(landmarks):
    # Landmarks indices:
    # 4 = thumb tip, 8 = index tip, 12 = middle tip, 16 = ring tip, 20 = pinky tip

    thumb = landmarks[4].y
    index = landmarks[8].y
    middle = landmarks[12].y
    ring = landmarks[16].y
    pinky = landmarks[20].y

    # Thumbs Up
    if thumb < index and middle > index and ring > index:
        return "THUMBS_UP"

    # Middle Finger (🖕)
    if middle < index and ring > middle and pinky > middle:
        return "MIDDLE_FINGER"

    # Open Palm
    if index < ring and middle < ring:
        return "OPEN_PALM"

    return "FIST"

if gesture_image:
    img = Image.open(gesture_image).convert("RGB")
    st.image(img, width=250)

    img_np = np.array(img)
    results = hands.process(img_np)

    if st.button("Detect Gesture"):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                gesture = detect_gesture(hand_landmarks.landmark)

                st.success(f"Detected Gesture: {gesture}")
                st.write("### Meaning:")
                st.write(GESTURE_MEANINGS.get(gesture, "Meaning not available"))
        else:
            st.error("No hand detected in image")
