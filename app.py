from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask_cors import CORS
import base64
import io
from PIL import Image
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the trained model
model = None
try:
    model = load_model("digit_model.keras")  # Try loading Keras model
    logging.info("✅ Model loaded successfully: digit_model.keras")
except Exception as e:
    logging.warning(f"⚠️ Error loading digit_model.keras: {e}")
    try:
        model = load_model("digit_model.h5")  # Fallback to HDF5 model
        logging.info("✅ Fallback: digit_model.h5 loaded successfully")
    except Exception as e:
        logging.error(f"❌ Critical Error: Unable to load model - {e}")
        model = None  # Ensure the model remains None if loading fails

@app.route('/')
def home():
    """Render the frontend page."""
    return render_template("index.html")  # Ensure 'index.html' is in the 'templates/' folder

@app.route('/predict', methods=['POST'])
def predict():
    """Handle digit prediction requests."""
    try:
        data = request.json
        if not data or "image" not in data:
            return jsonify({"error": "Missing 'image' key in request data"}), 400

        image_data = data["image"]  # Extract base64 image data

        # Convert base64 image to PIL image
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes)).convert("L")  # Convert to grayscale
        img = img.resize((28, 28))  # Resize to 28x28 pixels

        # Convert image to NumPy array
        img_array = np.array(img) / 255.0  # Normalize pixel values (0-1)
        img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for model input

        # Ensure model is loaded before making a prediction
        if model is None:
            return jsonify({"error": "Model failed to load. Check server logs"}), 500

        # Make prediction
        prediction = model.predict(img_array)
        predicted_digit = int(np.argmax(prediction))  # Get the predicted digit

        logging.info(f"🔢 Predicted digit: {predicted_digit}")
        return jsonify({"prediction": predicted_digit})

    except Exception as e:
        logging.error(f"❌ Prediction Error: {e}")
        return jsonify({"error": str(e)}), 500

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
  # Run on all network interfaces
