import os
import warnings
import tensorflow as tf
import numpy as np
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

CLASS_NAMES = ["cleanskin", "mild", "moderate", "severe", "unknown"]
IMG_SIZE = 224

MODEL_INFO = {
    "Classical CNN": "./models/classicalCNNmodel.keras",
    "EfficientNet": "./models/Effiecientnet_acne_classification_model.keras",
    "ResNet50": "./models/resnet50_acne_model.keras",
    "VGG": "./models/VisualGeometryGroup.keras"
}

TEST_DIR = os.path.expanduser("~/Desktop/modelbackup/dataset/test")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "heic"}

def custom_input_layer(*args, **kwargs):
    if "batch_shape" in kwargs:
        kwargs["batch_input_shape"] = kwargs.pop("batch_shape")
    return tf.keras.layers.InputLayer(*args, **kwargs)

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img).astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

models = {}
with tf.keras.utils.custom_object_scope({"InputLayer": custom_input_layer}):
    for name, path in MODEL_INFO.items():
        models[name] = tf.keras.models.load_model(path, compile=False)

for model_name, model in models.items():
    stats = {cls: {"correct": 0, "wrong": 0} for cls in CLASS_NAMES}

    for true_label in CLASS_NAMES:
        class_dir = os.path.join(TEST_DIR, true_label)
        if not os.path.isdir(class_dir):
            continue

        for file in os.listdir(class_dir):
            if not file.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
                continue

            img_path = os.path.join(class_dir, file)

            try:
                img = preprocess_image(img_path)
                preds = model.predict(img, verbose=0)[0]
                pred_label = CLASS_NAMES[np.argmax(preds)]

                if pred_label == true_label:
                    stats[true_label]["correct"] += 1
                else:
                    stats[true_label]["wrong"] += 1
            except:
                continue

    print(model_name)
    for cls in CLASS_NAMES:
        print(
            f"{cls:<10} -> correctly predicted: {stats[cls]['correct']} , "
            f"wrongly predicted: {stats[cls]['wrong']}"
        )
    print("-" * 50)
