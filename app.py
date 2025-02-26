import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend requests

# Load trained model
MODEL_PATH = "en_tokenizer.pkl"
model = load_model(MODEL_PATH)

# Image size (must match training size)
IMG_SIZE = 224  

# Function to preprocess the image
from flask import render_template

@app.route("/")
def home():
    return render_template("index.html")


def preprocess_image(img):
    img = img.convert("RGB")  # Ensure 3 channels
    img = img.resize((IMG_SIZE, IMG_SIZE))  # Resize
    img_array = img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array

# Prediction function
def predict_xray(img):
    class_labels = ["Benign", "Malignant", "Normal"]
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)  # Get probability scores for all classes
    predicted_class_index = np.argmax(predictions)  # Get highest probability class
    confidence = np.max(predictions)  # Confidence score
    predicted_class = class_labels[predicted_class_index]  # Get class name

    return {
        "result": predicted_class,
        "confidence": float(confidence)
    }

# Route for image upload & prediction
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    img = Image.open(file)
    
    # Get prediction
    prediction = predict_xray(img)
    
    return jsonify(prediction)

if __name__ == "__main__":
    app.run(debug=True)
