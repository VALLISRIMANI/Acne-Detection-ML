import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

CLASS_NAMES = ["cleanskin", "mild", "moderate", "severe", "unknown"]
IMG_SIZE = 224
BATCH_SIZE = 32

model = tf.keras.models.load_model(
    "../models/classicalCNNmodel.keras",
    custom_objects={"InputLayer": tf.keras.layers.InputLayer}
)

val_data = tf.keras.utils.image_dataset_from_directory(
    "../dataset/valid/",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int",
    class_names=CLASS_NAMES,
    color_mode="rgb",
    shuffle=False
)

val_data = val_data.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
val_data = val_data.ignore_errors()

y_true = []
y_pred = []

for x, y in val_data:
    preds = model.predict(x, verbose=0)
    y_true.extend(y.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

metrics = {
    "Accuracy": accuracy * 100,
    "Precision": precision * 100,
    "Recall": recall * 100,
    "F1-score": f1 * 100
}

print("Metrics (%):")
for k, v in metrics.items():
    print(f"{k}: {v:.2f}%")

plt.figure()
plt.bar(metrics.keys(), metrics.values())
plt.ylabel("Percentage")
plt.ylim(0, 100)

for i, v in enumerate(metrics.values()):
    plt.text(i, v + 1, f"{v:.0f}%", ha="center")

plt.title("Validation Metrics")
plt.show()
