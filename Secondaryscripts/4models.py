import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

IMG_SIZE = 224
BATCH_SIZE = 32
CLASS_NAMES = ["cleanskin", "mild", "moderate", "severe", "unknown"]

model = tf.keras.models.load_model(
    "../models/VisualGeometryGroup.keras",
    compile=False
)

X = []
y = []

base_dir = "test/test"

for label, class_name in enumerate(CLASS_NAMES):
    class_dir = os.path.join(base_dir, class_name)
    for file in os.listdir(class_dir):
        path = os.path.join(class_dir, file)
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img = np.array(img).astype("float32") / 255.0
            X.append(img)
            y.append(label)
        except:
            pass

X = np.array(X)
y_true = np.array(y)

y_pred = np.argmax(model.predict(X, batch_size=BATCH_SIZE), axis=1)

accuracy = np.mean(y_true == y_pred)
print("Accuracy:", round(accuracy * 100, 2), "%")

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    output_dict=True
)

overall_precision = report["weighted avg"]["precision"]
overall_recall = report["weighted avg"]["recall"]
overall_f1 = report["weighted avg"]["f1-score"]

print("\nOverall Metrics")
print("Precision:", round(overall_precision * 100, 2), "%")
print("Recall:", round(overall_recall * 100, 2), "%")
print("F1-score:", round(overall_f1 * 100, 2), "%")

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()

metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
values = [
    accuracy * 100,
    overall_precision * 100,
    overall_recall * 100,
    overall_f1 * 100
]

plt.figure()
plt.bar(metrics, values)
plt.ylabel("Percentage (%)")
plt.title("Overall Model Performance Metrics")
plt.ylim(0, 100)

for i, v in enumerate(values):
    plt.text(i, v + 1, f"{v:.2f}%", ha="center")

plt.show()
