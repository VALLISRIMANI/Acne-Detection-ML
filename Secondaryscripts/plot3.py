import matplotlib.pyplot as plt

# -------- Training history (from your logs) --------
train_acc = [
    0.3947, 0.5565, 0.6323, 0.6900, 0.7579,
    0.7922, 0.8395, 0.8631, 0.8882, 0.9057,
    0.9145, 0.9351, 0.9342, 0.9479, 0.9623,
    0.9587, 0.9624, 0.9715, 0.9731, 0.9668,
    0.9794, 0.9791, 0.9837, 0.9738, 0.9799
]

val_acc = [
    0.5608, 0.6318, 0.6498, 0.6847, 0.7028,
    0.6799, 0.6931, 0.6931, 0.7136, 0.7184,
    0.7341, 0.7401, 0.7581, 0.7304, 0.7449,
    0.7461, 0.7485, 0.7581, 0.7497, 0.7569,
    0.7581, 0.7425, 0.7521, 0.7377, 0.7497
]

train_loss = [
    1.3608, 1.0031, 0.8404, 0.7377, 0.6097,
    0.5247, 0.4304, 0.3708, 0.3047, 0.2454,
    0.2372, 0.1813, 0.1763, 0.1437, 0.1156,
    0.1252, 0.1188, 0.0934, 0.0901, 0.1003,
    0.0779, 0.0777, 0.0669, 0.0829, 0.0609
]

val_loss = [
    1.0838, 0.8738, 0.8625, 0.8193, 0.8225,
    1.0096, 0.9010, 0.8856, 0.9077, 0.9632,
    0.9540, 1.0772, 1.0533, 1.1416, 1.1533,
    1.2497, 1.1605, 1.2267, 1.3244, 1.3371,
    1.2201, 1.3101, 1.3205, 1.3173, 1.3174
]

epochs = range(1, len(train_acc) + 1)

# -------- Clean 2-in-1 figure --------
plt.figure(figsize=(12, 5))

# Accuracy subplot
plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, label="Training Accuracy")
plt.plot(epochs, val_acc, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)

# Loss subplot
plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, label="Training Loss")
plt.plot(epochs, val_loss, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
