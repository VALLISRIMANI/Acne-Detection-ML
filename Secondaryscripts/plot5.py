import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -------- Confusion Matrix values (from your results) --------
cm = np.array([
    [164, 10, 18, 4, 15],
    [7, 132, 51, 20, 1],
    [11, 38, 131, 28, 3],
    [4, 29, 37, 138, 3],
    [18, 1, 4, 1, 185]
])

class_names = ["cleanskin", "mild", "moderate", "severe", "unknown"]

# -------- Normalize (row-wise / true label) --------
cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

# -------- Plot Normalized Confusion Matrix --------
plt.figure(figsize=(7, 6))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_normalized,
    display_labels=class_names
)
disp.plot(cmap="Blues", values_format=".2f")

plt.title("Normalized Confusion Matrix (VGG Model)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
