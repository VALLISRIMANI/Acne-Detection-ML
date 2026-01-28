import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

CLASS_NAMES = ["cleanskin", "mild", "moderate", "severe", "unknown"]
IMG_SIZE = 224
BATCH_SIZE = 32

model = tf.keras.models.load_model(
    "../models/classicalCNNmodel.keras",
    custom_objects={"InputLayer": tf.keras.layers.InputLayer}
)

test_data = tf.keras.utils.image_dataset_from_directory(
    "../dataset/test/",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int",
    class_names=CLASS_NAMES,
    color_mode="rgb",
    shuffle=False
)

test_data = test_data.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
test_data = test_data.ignore_errors()

# Collect true & predicted labels
y_true = []
y_pred = []
total_per_class = {i: 0 for i in range(len(CLASS_NAMES))}
correct_per_class = {i: 0 for i in range(len(CLASS_NAMES))}

for x, y in test_data:
    preds = model.predict(x, verbose=0)
    pred_labels = np.argmax(preds, axis=1)
    y_true.extend(y.numpy())
    y_pred.extend(pred_labels)
    
    for true, pred in zip(y.numpy(), pred_labels):
        total_per_class[true] += 1
        if true == pred:
            correct_per_class[true] += 1

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Calculate test metrics
test_accuracy = accuracy_score(y_true, y_pred) * 100
test_precision = precision_score(y_true, y_pred, average="weighted", zero_division=0) * 100
test_recall = recall_score(y_true, y_pred, average="weighted", zero_division=0) * 100
test_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100

# Prepare data for plotting
correct = [correct_per_class[i] for i in range(len(CLASS_NAMES))]
wrong = [total_per_class[i] - correct_per_class[i] for i in range(len(CLASS_NAMES))]
total_images = sum(total_per_class.values())
total_correct = sum(correct)
overall_accuracy = (total_correct / total_images) * 100

# ---------- PLOT ----------
fig, ax = plt.subplots(figsize=(12, 8))

x = np.arange(len(CLASS_NAMES))
ax.bar(x, correct, label="Correct", color='#2ecc71')
ax.bar(x, wrong, bottom=correct, label="Wrong", color='#e74c3c')

ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, rotation=20)
ax.set_ylabel("Number of Images", fontsize=12)
ax.set_title("Test Dataset Performance - Correct vs Wrong Predictions", fontsize=14, fontweight='bold')
ax.legend()

# Add count labels on bars
for i in range(len(CLASS_NAMES)):
    if correct[i] > 0:
        ax.text(i, correct[i] / 2, str(correct[i]), ha="center", va="center", color="white", fontweight='bold')
    if wrong[i] > 0:
        ax.text(i, correct[i] + wrong[i] / 2, str(wrong[i]), ha="center", va="center", color="white", fontweight='bold')

# ---------- TABLE DATA ----------
table_data = []
for i, name in enumerate(CLASS_NAMES):
    acc = (correct[i] / total_per_class[i]) * 100 if total_per_class[i] > 0 else 0
    table_data.append([
        name,
        total_per_class[i],
        correct[i],
        wrong[i],
        f"{acc:.2f}%"
    ])

table_data.append([
    "OVERALL",
    total_images,
    total_correct,
    total_images - total_correct,
    f"{overall_accuracy:.2f}%"
])

# ---------- DRAW TABLE ----------
table = ax.table(
    cellText=table_data,
    colLabels=["Class", "Total", "Correct", "Wrong", "Accuracy"],
    loc="bottom",
    cellLoc="center",
    colWidths=[0.15, 0.1, 0.1, 0.1, 0.12]
)
table.scale(1, 1.5)
table.auto_set_font_size(False)
table.set_fontsize(9)

# Add metrics text box
metrics_text = (
    f"TEST DATASET METRICS (Weighted)\n"
    f"{'='*40}\n"
    f"Accuracy  : {test_accuracy:.2f}%\n"
    f"Precision : {test_precision:.2f}%\n"
    f"Recall    : {test_recall:.2f}%\n"
    f"F1-Score  : {test_f1:.2f}%\n"
    f"{'='*40}\n"
    f"Computed on TEST dataset (unseen data)"
)

ax.text(0.98, 0.97, metrics_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        family='monospace')

plt.subplots_adjust(left=0.1, bottom=0.35, right=0.98, top=0.95)
plt.tight_layout()
plt.show()