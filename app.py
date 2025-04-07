from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask_cors import CORS
import base64
import io
from PIL import Image
import logging
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load model
model = None
try:
    model = load_model("digit_model.h5")
    logging.info("✅ Model loaded successfully: digit_model.h5")
except Exception as e:
    logging.error(f"❌ Error loading model: {e}")
    model = None

@app.route("/")
def home():
    """Serve the frontend page."""
    return render_template("index.html")  # Make sure 'index.html' is inside a 'templates/' folder

@app.route("/predict", methods=["POST"])
def predict():
    """Handle digit prediction."""
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Missing image data"}), 400

        # Decode base64 image
        image_data = base64.b64decode(data["image"])
        img = Image.open(io.BytesIO(image_data)).convert("L")
        img = img.resize((28, 28))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        # Predict
        prediction = model.predict(img_array)
        predicted_digit = int(np.argmax(prediction))
        logging.info(f"🔢 Predicted digit: {predicted_digit}")
        return jsonify({"prediction": predicted_digit})

    except Exception as e:
        logging.error(f"❌ Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
