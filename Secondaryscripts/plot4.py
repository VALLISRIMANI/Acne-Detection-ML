correct = {
    "cleanskin": [164],
    "mild": [132],
    "moderate": [131],
    "severe": [ 138],
    "unknown": [ 185]
}

wrong = {
    "cleanskin": [47],
    "mild": [79],
    "moderate": [ 80],
    "severe": [73],
    "unknown": [ 24]
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

plt.xticks(x + width * 2, "VGG", rotation=15)
plt.ylabel("Prediction Count")
plt.title("Model-wise Correct vs Wrong Predictions (All Classes)")
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.show()
