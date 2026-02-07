import os
import io
import warnings
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
from pymongo import MongoClient, ReturnDocument
import cloudinary
import cloudinary.uploader

from dotenv import load_dotenv
load_dotenv()


register_heif_opener()

app = Flask(__name__)

cloudinary.config(
    cloud_name=os.getenv("CLOUD_NAME"),
    api_key=os.getenv("API_KEY"),
    api_secret=os.getenv("API_SECRET"),
    secure=True
)

client = MongoClient(os.getenv("MONGODB_URI"))
db = client["acne_ai"]
predictions_col = db["predictions"]
counters_col = db["counters"]

if counters_col.count_documents({"_id": "prediction_id"}) == 0:
    counters_col.insert_one({"_id": "prediction_id", "seq": 0})

def get_next_id():
    doc = counters_col.find_one_and_update(
        {"_id": "prediction_id"},
        {"$inc": {"seq": 1}},
        return_document=ReturnDocument.AFTER
    )
    return doc["seq"]

def custom_input_layer(*args, **kwargs):
    if "batch_shape" in kwargs:
        kwargs["batch_input_shape"] = kwargs.pop("batch_shape")
    return tf.keras.layers.InputLayer(*args, **kwargs)

with tf.keras.utils.custom_object_scope({"InputLayer": custom_input_layer}):
    model = tf.keras.models.load_model("./models/VisualGeometryGroup.keras")

CLASS_NAMES = ["cleanskin", "mild", "moderate", "severe", "unknown"]
IMG_SIZE = 224
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "heif", "heic", "avif"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Acne Severity Classification API"})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "image file is required"}), 400

    try:
        file = request.files["image"]

        if file.filename == "":
            return jsonify({"error": "empty filename"}), 400

        if not allowed_file(file.filename):
            return jsonify({
                "error": "unsupported file type",
                "allowed": list(ALLOWED_EXTENSIONS)
            }), 400

        image = Image.open(file)

        img = preprocess_image(image)
        probs = model.predict(img)[0]

        idx = int(np.argmax(probs))
        prediction = CLASS_NAMES[idx]
        confidence = float(round(float(probs[idx]) * 100, 2))

        prediction_id = get_next_id()

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)

        public_id = f"acne_dataset/{prediction}/{prediction}_{prediction_id}"

        upload_result = cloudinary.uploader.upload(
            buffer,
            public_id=public_id,
            overwrite=False,
            resource_type="image"
        )

        image_url = upload_result["secure_url"]

        probabilities = {
            CLASS_NAMES[i]: float(round(float(probs[i]) * 100, 2))
            for i in range(len(CLASS_NAMES))
        }

        doc = {
            "prediction_id": prediction_id,
            "image_url": image_url,
            "cloudinary_public_id": public_id,
            "predicted_class": prediction,
            "confidence": confidence,
            "probabilities": probabilities,
            "created_at": datetime.utcnow()
        }

        predictions_col.insert_one(doc)

        return jsonify({
            "prediction_id": prediction_id,
            "image_url": image_url,
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
