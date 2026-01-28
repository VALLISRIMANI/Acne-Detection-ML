import matplotlib.pyplot as plt
import numpy as np

models = ["Classical CNN", "EfficientNet", "ResNet50", "VGG"]

correct = {
    "cleanskin": [106, 160, 165, 164],
    "mild": [23, 154, 96, 132],
    "moderate": [47, 86, 58, 131],
    "severe": [183, 124, 86, 138],
    "unknown": [199, 181, 144, 185]
}

wrong = {
    "cleanskin": [105, 51, 46, 47],
    "mild": [188, 57, 115, 79],
    "moderate": [164, 125, 153, 80],
    "severe": [28, 87, 125, 73],
    "unknown": [10, 28, 65, 24]
}

colors = {
    "cleanskin": "pink",
    "mild": "orange",
    "moderate": "red",
    "severe": "purple",
    "unknown": "blue"
}

x = np.arange(len(models))
width = 0.15

plt.figure(figsize=(14, 7))

for i, cls in enumerate(colors.keys()):
    plt.bar(
        x + i * width,
        correct[cls],
        width,
        color=colors[cls],
        label=f"{cls} correct"
    )
    plt.bar(
        x + i * width,
        wrong[cls],
        width,
        bottom=correct[cls],
        color=colors[cls],
        alpha=0.4,
        label=f"{cls} wrong"
    )

plt.xticks(x + width * 2, models, rotation=15)
plt.ylabel("Prediction Count")
plt.title("Model-wise Correct vs Wrong Predictions (All Classes)")
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.show()
