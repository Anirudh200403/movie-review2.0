from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
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

# Load the digit recognition model
model = None
try:
    model = load_model("digit_model.keras")  # Primary model (if available)
    logging.info("✅ Model loaded: digit_model.keras")
except Exception as e:
    logging.warning(f"⚠️ Couldn't load digit_model.keras: {e}")
    try:
        model = load_model("digit_model.h5")  # Fallback model
        logging.info("✅ Fallback model loaded: digit_model.h5")
    except Exception as e:
        logging.error(f"❌ Failed to load any model: {e}")
        model = None

# Home route (renders index.html)
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded."}), 500

    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Missing 'image' in request"}), 400

        # Decode base64 image
        image_data = base64.b64decode(data["image"])
        img = Image.open(io.BytesIO(image_data)).convert("L")  # Grayscale
        img = img.resize((28, 28))  # Resize to match model input
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # Predict
        prediction = model.predict(img_array)
        predicted_digit = int(np.argmax(prediction))
        logging.info(f"🔢 Prediction: {predicted_digit}")

        return jsonify({"prediction": predicted_digit})
    except Exception as e:
        logging.error(f"❌ Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

# Main entry for local and Render use
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render provides $PORT
    app.run(host="0.0.0.0", port=port)
