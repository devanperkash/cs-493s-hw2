import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Configuration
ops = ['+', '-']
primes = [97, 113]
layers_list = [1, 2]
seeds = [1, 2, 3]
log_dir = "logs"
out_dir = "plots"
os.makedirs(out_dir, exist_ok=True)

# Operator symbol to filename-safe string
op_str_map = {'+': 'plus', '-': 'minus'}

# Load metrics from JSON
def load_metrics(op_str, p, layers, seed):
    path = f"{log_dir}/log_{op_str}_p{p}_{layers}layer_seed{seed}.json"
    if not os.path.exists(path):
        print(f"Missing: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)

# Group metrics and plot
for op in ops:
    op_str = op_str_map[op]
    for p in primes:
        for layers in layers_list:
            metrics = [load_metrics(op_str, p, layers, seed) for seed in seeds]
            metrics = [m for m in metrics if m is not None]
            if not metrics:
                continue  # Skip if no data

            # Aggregate
            train_losses = np.array([m["train_loss"] for m in metrics])
            val_accuracies = np.array([m["val_accuracy"] for m in metrics])
            steps = [1000 * (i + 1) for i in range(train_losses.shape[1])]

            avg_train_loss = train_losses.mean(axis=0)
            avg_val_acc = val_accuracies.mean(axis=0)

            # Plot Training Loss
            plt.figure()
            plt.plot(steps, avg_train_loss, label="Train Loss")
            plt.xlabel("Training Steps")
            plt.ylabel("Loss")
            plt.title(f"Train Loss: {op} mod {p}, {layers} layer(s)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{out_dir}/train_{op_str}_p{p}_{layers}layer.png")
            plt.close()

            # Plot Validation Accuracy
            plt.figure()
            plt.plot(steps, avg_val_acc, label="Val Accuracy")
            plt.xlabel("Training Steps")
            plt.ylabel("Accuracy")
            plt.title(f"Validation Accuracy: {op} mod {p}, {layers} layer(s)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{out_dir}/val_{op_str}_p{p}_{layers}layer.png")
            plt.close()