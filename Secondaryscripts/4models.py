import os
import warnings
import base64
import io

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

CLASS_NAMES = ["cleanskin", "mild", "moderate", "severe", "unknown"]
IMG_SIZE = 224

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "heic"}

MODEL_INFO = {
    "Classical CNN": "./models/classicalCNNmodel.keras",
    "EfficientNet": "./models/Effiecientnet_acne_classification_model.keras",
    "ResNet50": "./models/resnet50_acne_model.keras",
    "VGG": "./models/VisualGeometryGroup.keras"
}

def custom_input_layer(*args, **kwargs):
    if "batch_shape" in kwargs:
        kwargs["batch_input_shape"] = kwargs.pop("batch_shape")
    return tf.keras.layers.InputLayer(*args, **kwargs)

models = {}
with tf.keras.utils.custom_object_scope({"InputLayer": custom_input_layer}):
    for name, path in MODEL_INFO.items():
        models[name] = tf.keras.models.load_model(path)

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image).astype("float32") / 255.0
    return np.expand_dims(image, axis=0)

def is_allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    all_results = []

    if request.method == "POST":
        images = request.files.getlist("images")

        for img_file in images:
            if not img_file or not is_allowed(img_file.filename):
                continue

            try:
                image = Image.open(img_file)

                buffer = io.BytesIO()
                image.save(buffer, format="JPEG")
                img_base64 = base64.b64encode(buffer.getvalue()).decode()

                img = preprocess_image(image)

                model_results = []
                for name, model in models.items():
                    probs = model.predict(img, verbose=0)[0]
                    idx = np.argmax(probs)

                    model_results.append({
                        "model": name,
                        "prediction": CLASS_NAMES[idx],
                        "confidence": round(float(probs[idx]) * 100, 2)
                    })

                all_results.append({
                    "image": img_base64,
                    "results": model_results
                })

            except Exception:
                continue

    return render_template("index.html", all_results=all_results)

if __name__ == "__main__":
    app.run(debug=False)
