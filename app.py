import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained DenseNet model
MODEL_PATH = "xray_densenet_model2.h5"
model = load_model(MODEL_PATH)

# Define image size (must match training size)
IMG_SIZE = 224  

# Function to preprocess the uploaded image
from tensorflow.keras.preprocessing.image import img_to_array  # Import here


def preprocess_image(img):
    img = img.convert("RGB")  # Ensure 3 channels (RGB)
    img = img.resize((IMG_SIZE, IMG_SIZE))  # Resize
    img_array = img_to_array(img)  # Convert to NumPy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array


# def preprocess_image(img_path):
#     img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))  # Load and resize
#     img_array = image.img_to_array(img)  # Convert to array
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#     img_array = img_array / 255.0  # Normalize
#     return img_array


# Function to make predictions
def predict_xray(img):
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)[0][0]  # Get prediction score
  
    
    if prediction >= 0.5:
        return "✅ Benign (Non-Cancerous)", 1 - prediction
    else:
        return "⚠️ Malignant (Cancerous)", 1 - prediction

# Streamlit UI Design
st.set_page_config(page_title="Lung Cancer Detection", page_icon="🩺", layout="centered")

st.title("🩺 Lung Cancer Detection")
st.write("Upload an X-ray or CT scan to check for **malignant (cancerous)** or **benign (non-cancerous)** tumors.")

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
    else:
        st.success("✅ No signs of malignancy detected, but regular checkups are advised.")

st.write("🔬 Model powered by **DenseNet121** | Developed by Rohit 🚀")
