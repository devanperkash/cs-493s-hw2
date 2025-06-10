import json
import matplotlib.pyplot as plt

# Replace this with your actual file path
log_path = "logs/grok_div_p97_1layer_seed1_20250609_234029.json"

with open(log_path, "r") as f:
    metrics = json.load(f)

epochs = metrics["epoch"]
train_loss = metrics["train_loss"]
val_acc = metrics["val_accuracy"]

# Plot Training Loss
plt.figure()
plt.plot(epochs, train_loss, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train Loss")
plt.grid(True)
plt.legend()
plt.show()

# Plot Validation Accuracy
plt.figure()
plt.plot(epochs, val_acc, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")
plt.grid(True)
plt.legend()
plt.show()