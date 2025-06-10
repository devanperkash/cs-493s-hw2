import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

ops = ['+', '-']
primes = [97, 113]
layers_list = [1, 2]
seeds = [1, 2, 3]
log_dir = "logs"
out_dir = "plots"
os.makedirs(out_dir, exist_ok=True)

def load_metrics(op, p, layers, seed):
    path = f"{log_dir}/log_{op}_p{p}_{layers}layer_seed{seed}.json"
    with open(path, "r") as f:
        return json.load(f)

# Group metrics for each config
for op in ops:
    for p in primes:
        for layers in layers_list:
            train_losses = []
            val_accuracies = []

            for seed in seeds:
                m = load_metrics(op, p, layers, seed)
                train_losses.append(m["train_loss"])
                val_accuracies.append(m["val_accuracy"])

            # Convert to numpy arrays for averaging
            train_losses = np.array(train_losses)
            val_accuracies = np.array(val_accuracies)
            epochs = list(range(1, len(train_losses[0]) + 1))

            # Average over seeds
            avg_train_loss = train_losses.mean(axis=0)
            avg_val_acc = val_accuracies.mean(axis=0)

            # Plot Training Loss
            plt.figure()
            plt.plot(epochs, avg_train_loss, label="Train Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Train Loss: {op} mod {p}, {layers} layer(s)")
            plt.grid(True)
            plt.savefig(f"{out_dir}/train_{op}_p{p}_{layers}layer.png")
            plt.close()

            # Plot Validation Accuracy
            plt.figure()
            plt.plot(epochs, avg_val_acc, label="Val Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"Validation Accuracy: {op} mod {p}, {layers} layer(s)")
            plt.grid(True)
            plt.savefig(f"{out_dir}/val_{op}_p{p}_{layers}layer.png")
            plt.close()