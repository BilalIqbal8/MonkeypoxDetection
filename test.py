import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd
import os

# Set page config
st.set_page_config(
    page_title="Monkeypox Detection System",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #000000;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .main {
        background-color: #000000;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    .stButton>button, .nav-button {
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        margin-right: 10px;
        text-decoration: none;
        display: inline-block.
    }
    .stButton>button:hover, .nav-button:hover {
        background-color: #0056b3.
    }
    .stFileUploader {
        border: 2px dashed #ccc.
        border-radius: 5px.
        padding: 10px.
        margin-bottom: 20px.
    }
    .stTextInput, .stNumberInput {
        border: 1px solid #ccc.
        border-radius: 5px.
        padding: 10px.
        width: 100%.
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load pre-trained model
@st.cache_resource()
def load_model():
        model_path = 'C:\\Users\Adil\\Downloads\\monkeypox_model.keras'
        
        if not os.path.isfile(model_path):
            st.error(f"Model file not found at: {model_path}")
            return None

        try:
            model = tf.keras.models.load_model(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
        return model

model = load_model()

# Check if model loaded successfully
if model is None:
    st.stop()

# Preprocess image
def preprocess_image(image):
    image = image.resize((256, 256))  # Assuming the model expects 256x256 input
    image = np.array(image)
    if image.shape[2] == 4:  # Convert RGBA to RGB if necessary
        image = image[..., :3]
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)
    return image

# Make prediction
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

# Initialize patient records DataFrame
if 'patient_records' not in st.session_state:
    st.session_state.patient_records = pd.DataFrame(columns=["Name", "Age", "Gender", "Result"])

# Navigation
with st.sidebar:
    selected = option_menu("Navigation", ["Home", "Symptoms", "Precautions", "Medication", "Patient Records", "About"],
                           icons=['house', 'clipboard-heart', 'shield-check', 'capsule', 'clipboard-data', 'info-circle'],
                           menu_icon="cast", default_index=0, orientation="vertical")

# Header with image
st.markdown(
    """
    <div class="header">
       <img src="https://news.cuanschutz.edu/hubfs/School%20of%20Medicine/Monkey%20Pox%20-%208-8-22.png" 
     alt="Logo" 
     style="width: 100%; height: 300px; object-fit: cover;">
        
    </div>
    """,
    unsafe_allow_html=True
)

# Home page content
if selected == "Home":
    st.write('Upload an image to test for Monkeypox and fill in your details.')

    # Patient Details Form
    with st.form("patient_form"):
        name = st.text_input("Name", placeholder="Enter your name")
        age = st.number_input("Age", min_value=0, max_value=120, step=1, value=0)
        gender = st.radio("Gender", ["Male", "Female", "Other"])
        uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])
        submit_button = st.form_submit_button(label='Submit')

        if submit_button and uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.write('Classifying...')
            
            try:
                prediction = predict(image)
                result = "Monkeypox Positive" if prediction[0][0] < 0.5 else "Monkeypox Negative"
                st.success(f'The model predicts: {result}')
                
                # Store patient details and result
                new_record = pd.DataFrame([[name, age, gender, result]], columns=["Name", "Age", "Gender", "Result"])
                st.session_state.patient_records = pd.concat([st.session_state.patient_records, new_record], ignore_index=True)
            except Exception as e:
                st.error(f"Error predicting image: {e}")

elif selected == "Symptoms":
    st.title('Monkeypox Symptoms')

    symptoms = [
        ("1)Fever", "https://assets.clevelandclinic.org/transform/LargeFeatureImage/6323bdcd-dcee-4a1d-b3ba-9200715d94c5/bodyHappenFever-1006577818-770x553_jpg"),
        ("2)Headache", "https://emedmultispecialtygroup.com/wp-content/uploads/2023/09/tension_headache_300x300.jpg"),
        ("3)Muscle Aches", "https://di.myupchar.com/1968/muscle-ache-myalgia-maspeshiyo-me-dard-ke-lakshan-karan-upchar-bachav-ilaj-dawa-in-hindi.webp"),
        ("4)Backache", "https://cdn.5280.com/2014/03/aching.jpg"),
        ("5)Swollen Lymph Nodes", "https://thefootcaregroup.co.uk/wp-content/uploads/2023/05/swollen-feet-scaled.jpg"),
        ("6)Chills", "https://images.healthshots.com/healthshots/en/uploads/2023/12/21120716/chills.jpg"),
        ("7)Exhaustion", "https://coloradopaincare.com/wp-content/uploads/2018/07/colorado-pain-care-exhausted-blog.jpg"),
        ("8)Rash", "https://images.theconversation.com/files/209558/original/file-20180308-30983-e4u830.jpg?ixlib=rb-4.1.0&q=20&auto=format&w=320&fit=clip&dpr=2&usm=12&cs=strip"),
    ]

    for symptom, image_url in symptoms:
        st.subheader(symptom)
        st.image(image_url, width=400)  # Set image width to 400

elif selected == "Precautions":
    st.title('Monkeypox Precautions')
    st.write("""
        To prevent the spread of Monkeypox, follow these precautions:
        - Avoid contact with animals that could harbor the virus.
        - Avoid contact with any materials, such as bedding, that have been in contact with a sick animal.
        - Isolate infected patients from others who could be at risk.
        - Practice good hand hygiene with soap and water or use an alcohol-based sanitizer.
        - Use personal protective equipment (PPE) when caring for patients.
    """)

elif selected == "Medication":
    st.title('Monkeypox Medication')

    medications = [
        ("Antiviral drugs (like Tecovirimat)", "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ6X5klF6_vYUPngTnd_y2DtTQlqtvGkoyeUg&s"),
        ("Vaccinia immune globulin (VIG)", "https://www.statnews.com/wp-content/uploads/2022/08/GettyImages-1412754952-1024x576.jpg"),
        ("Supportive care (hydration, pain management, treatment of secondary bacterial infections)", "https://ch-api.healthhub.sg/api/public/content/c009a34c840b40ee88bb46796eef161f?v=5ce8d779&t=azheaderimage"),
    ]

    for medication, image_url in medications:
        st.markdown(f"### {medication}")
        st.image(image_url, width=400)  # Set image width to 400

elif selected == "Patient Records":
    st.title('Patient Records')
    st.write('Here are the details of all the patients who have used the system.')
    st.dataframe(st.session_state.patient_records)

elif selected == "About":
    st.title('About Monkeypox Detection System')
    st.write("""
        The Monkeypox Detection System is designed to help detect Monkeypox using image analysis. 
        This system leverages a pre-trained deep learning model to analyze images and provide a prediction on whether the image indicates a Monkeypox infection. 
        The aim is to provide a quick and accessible tool for preliminary diagnosis, especially useful in areas with limited access to healthcare facilities.
    """)