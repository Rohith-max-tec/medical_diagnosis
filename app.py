import os
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Define model path inside the 'medical_diagnosis' folder
MODEL_PATH = "/app/xray_densenet_model6.h5"
# Check if the model file exists
#if not os.path.exists(MODEL_PATH):
    #raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# Load the trained DenseNet model
model = load_model(MODEL_PATH)

# Define image size (must match training size)
IMG_SIZE = 224  

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = img.convert("RGB")  # Ensure 3 channels (RGB)
    img = img.resize((IMG_SIZE, IMG_SIZE))  # Resize
    img_array = img_to_array(img)  # Convert to NumPy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array



# Function to make predictions
# def predict_xray(img):
#     img_array = preprocess_image(img)
#     prediction = model.predict(img_array)[0][0]  # Get prediction score
  
#     if prediction >= 0.5:
#         return "✅ Benign (Non-Cancerous)", 1 - prediction
#     else:
#         return "⚠️ Malignant (Cancerous)", 1 - prediction


def predict_xray(img_path):
    class_labels = [ "Benign","Malignant", "Normal"]
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array) # Get probability scores for all 3 classes
    predicted_class_index = np.argmax(predictions) # Get the index of the highest probability
    confidence = np.max(predictions)  # Get confidence score for the predicted class
    predicted_class = class_labels[predicted_class_index]  # Get the corresponding class name

    if predicted_class == "Benign":
        return "✅ Benign (Non-Cancerous)", confidence:.4f
    elif predicted_class == "Malignant":
        return "⚠️ Malignant (Cancerous)", confidence:.4f
    elif predicted_class == "Normal":
        return " Normal (You are healthy)", confidence:.4f
        
    
    

# Streamlit UI Design
st.set_page_config(page_title="Lung Cancer Detection", page_icon="🩺", layout="centered")

st.title("🩺 Lung Cancer Detection")
st.write("Upload an CT scan copy of your lungs to check for **malignant (cancerous)** or **benign (non-cancerous)** tumors.")

# File uploader
uploaded_file = st.file_uploader("Upload an X-ray or CT Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)  # Open the uploaded image
    st.image(image, caption="Uploaded X-ray Image", use_column_width=True)  # Display image
    
    st.write("🔍 **Processing Image...**")
    
    # Make prediction
    result, confidence = predict_xray(image)
    
    # Display result
    st.subheader(result)
    st.write(f"**Confidence Score:** {confidence:.4f}")

    # Additional health advice
    if "Malignant" in result:
        st.warning("⚠️ Please consult a doctor for further medical diagnosis.")
    elif  "Benign" in result:
        st.success("✅ No signs of malignancy detected, but regular checkups are advised.")

    elif  "Normal" in result:
        st.success("✅ No signs of malignancy detected, its normal.")


    

st.write(" **Note** : This model is still training. Please consult a doctor in case of an emergency or inaccurate.")
