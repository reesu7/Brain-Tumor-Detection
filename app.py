import streamlit as st
import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('BrainTumor.h5')

def predict_tumor(image):
    img = image.resize((64, 64))
    img = np.array(img)
    input_img = np.expand_dims(img, axis=0)
    result = model.predict(input_img)
    result_final = np.argmax(result, axis=1)
    return result_final

st.title("Brain Tumor Detection Using MRI Images")

upload_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])
button = st.button("Predict Tumor")

if button:
    if upload_file is not None:
        image = Image.open(upload_file)
        st.image(image, caption='Uploaded MRI Image', use_column_width=True)
        result_final = predict_tumor(image)
        if result_final == 0:
            st.success("BRAIN TUMOR NOT DETECTED")
        else:
            st.error("BRAIN TUMOR DETECTED")
    else:
        st.write("Please upload an MRI image.")
