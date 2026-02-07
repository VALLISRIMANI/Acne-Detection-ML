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
from huggingface_hub import hf_hub_download
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

def get_db():
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI not set")

    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    return client["acne_ai"]

def init_counter():
    db = get_db()
    counters_col = db["counters"]
    if counters_col.count_documents({"_id": "prediction_id"}) == 0:
        counters_col.insert_one({"_id": "prediction_id", "seq": 0})

def get_next_id():
    db = get_db()
    counters_col = db["counters"]
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

MODEL_PATH = hf_hub_download(
    repo_id="sivanagu1206/acnedetectionmodel",
    filename="VisualGeometryGroup.keras"
)

with tf.keras.utils.custom_object_scope({"InputLayer": custom_input_layer}):
    model = tf.keras.models.load_model(MODEL_PATH)

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
        init_counter()

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

        db = get_db()
        predictions_col = db["predictions"]
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