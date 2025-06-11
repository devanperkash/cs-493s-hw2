import json
import os
import numpy as np

ops = ['+', '-']
op_str_map = {'+': 'plus', '-': 'minus'}
primes = [97, 113]
layers_list = [1, 2]
seeds = [1, 2, 3]

log_dir = "logs"

print(f"{'Op':^5} {'p':^5} {'L':^7} {'Avg Final Train Loss':^25} {'Avg Final Val Acc':^25}")
print("-" * 70)

for op in ops:
    op_str = op_str_map[op]
    for p in primes:
        for l in layers_list:
            final_train_losses = []
            final_val_accuracies = []

            for seed in seeds:
                path = f"{log_dir}/log_{op_str}_p{p}_{l}layer_seed{seed}.json"
                if not os.path.exists(path):
                    print(f"Missing log file: {path}")
                    continue
                with open(path, 'r') as f:
                    data = json.load(f)
                    final_train_losses.append(data["train_loss"][-1])
                    final_val_accuracies.append(data["val_accuracy"][-1])

            # Compute averages
            avg_loss = np.mean(final_train_losses)
            avg_acc = np.mean(final_val_accuracies)

            print(f"{op:^5} {p:^5} {l:^7} {avg_loss:^25.4f} {avg_acc:^25.4f}")