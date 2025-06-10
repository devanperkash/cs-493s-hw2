import os
import subprocess

ops = ['+', '-']
primes = [97, 113]
layers = [1, 2]
seeds = [1, 2, 3]

for op in ops:
    for p in primes:
        for layer in layers:
            for seed in seeds:
                print(f"\n Running: op={op}, p={p}, layers={layer}, seed={seed}")
                cmd = [
                    "python", "train_mod.py",
                    "--op", op,
                    "--p", str(p),
                    "--layers", str(layer),
                    "--seed", str(seed),
                    "--steps", "100000"
                ]
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Failed run: {e}")