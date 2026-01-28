import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

def custom_input_layer(*args, **kwargs):
    if "batch_shape" in kwargs:
        kwargs["batch_input_shape"] = kwargs.pop("batch_shape")
    return tf.keras.layers.InputLayer(*args, **kwargs)

with tf.keras.utils.custom_object_scope({"InputLayer": custom_input_layer}):
    model = tf.keras.models.load_model("./models/VisualGeometryGroup.keras")

CLASS_NAMES = ["cleanskin", "mild", "moderate", "severe", "unknown"]
IMG_SIZE = 224

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image).astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "image file is required"}), 400

    try:
        image = Image.open(request.files["image"])
        img = preprocess_image(image)
        probs = model.predict(img)[0]

        idx = np.argmax(probs)
        prediction = CLASS_NAMES[idx]
        confidence = probs[idx] * 100

        return jsonify({
            "prediction": str(prediction),
            "confidence": float(round(confidence, 2)),
            "probabilities": {
                CLASS_NAMES[i]: float(round(float(probs[i]) * 100, 2))
                for i in range(len(CLASS_NAMES))
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
