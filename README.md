# Advanced ML HW 2: Modular Arithmetic with Transformers

This codebase is our work for the HW 2 assignment in CSE 599S (Advanced Machine Learning). The goal is to train and evaluate small transformer (GPT-style) models to perform modular arithmetic (addition, subtraction, division) over finite fields, exploring generalization and grokking phenomena.

**Note:** Only `model.py` was provided as starter code. All other code is original work.

## Project Structure

```
cs-493s-hw2/
  model.py                # Starter code: GPT model definition (provided)
  train_mod.py            # Main training script for modular arithmetic tasks
  train_grok.py           # Script for grokking experiments
  train_all_mods.py       # Batch training for all ops, moduli, layers, seeds
  generate_data.py        # Script to generate datasets for all tasks
  inference.py            # Script for model inference/generation
  plot_curves.py          # Plotting training/validation curves
  plot_grok.py            # Plotting for grokking runs
  summarize_metrics.py    # Summarize results across runs
  utils.py                # Data utilities (vocab, encoding, dataset)
  data/                   # Generated datasets (train/val/test splits)
  checkpoints/            # Saved model checkpoints
  logs/                   # Training logs and metrics
  plots/                  # Output plots
  Advanced_ML_HW_2.pdf    # Assignment specification
```

## Setup

1. **Install dependencies**  
   (Requires Python 3.10+ and PyTorch)
   ```bash
   pip install torch matplotlib numpy
   ```

2. **Generate data**  
   To (re)generate all modular arithmetic datasets:
   ```bash
   python generate_data.py
   ```

## Training

- **Train a model to reproduce the phrase "I love machine learning.":**
  ```bash
  python train_mod.py --op + --p 97 --layers 1 --seed 1 --steps 100000
  ```
  - `--op`   : Operation (`+`, `-`, `/`)
  - `--p`    : Prime modulus (`97` or `113`)
  - `--layers`: Number of transformer layers (`1` or `2`)
  - `--seed` : Random seed
  - `--steps`: Training steps (default: 100000)

- **Batch train all combinations of models on modular arithmetic:**
  ```bash
  python train_all_mods.py
  ```

- **Grokking experiments:**  
  Use `train_grok.py` for long training on small data fractions.

## Evaluation & Plotting

- **Plot learning curves:**
  ```bash
  python plot_curves.py
  ```
  Plots are saved in the `plots/` directory.

- **Summarize metrics of log files:**
  ```bash
  python summarize_metrics.py
  ```

- **Inference on simple "I love machine learning" model:**
  ```bash
  python inference.py
  ```

- **Generate data for modular arithmetic**
    ```bash
    python generate_data.py
    ```

## Notes

- All training/evaluation scripts auto-save checkpoints and logs.
- You can modify hyperparameters and experiment settings via script arguments.
