import streamlit as st
import numpy as np
import cv2
from keras.models import load_model

# Load the pre-trained model
model = load_model("monkeypox_model.keras")

# Define the labels for the binary classification
labels = [ "Monkey Pox","Non Monkey Pox"]

# Set up the Streamlit app
st.title("Monkey Pox Detection")
st.write("Upload an image to see if it contains Monkey Pox")

# Allow the user to upload an image
file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# Make a prediction if an image is uploaded
if file is not None:
    # Read the image and preprocess it
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_disp = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # For display purposes only
    img = cv2.resize(img, (256,256))
    img = np.expand_dims(img, axis=0) / 255.0

    # Make a prediction and display the result
    pred = model.predict(img)[0][0]
    label = labels[int(round(pred))]
    st.write("Prediction: " + label)

    # Display the original image
    st.write("Original Image:")
    st.image(img_disp, channels="BGR")
    
