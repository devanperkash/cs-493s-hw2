import torch
import torch.nn.functional as F
from model import GPT, GPTConfig
from utils import create_vocab, encode, decode, ModData
from torch.utils.data import DataLoader
import os
import random
import argparse
import json

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--op', type=str, required=True, choices=['+', '-', '/'])
parser.add_argument('--p', type=int, required=True, choices=[97, 113])
parser.add_argument('--layers', type=int, required=True, choices=[1, 2])
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--steps', type=int, default=100000)
args = parser.parse_args()

if __name__ == "__main__":
    # Set seed for reproducibility
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Device configuration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")

    # Parameters
    data_path = f"data/{args.op}_p{args.p}_train.txt"
    val_path = f"data/{args.op}_p{args.p}_val.txt"
    block_size = 16
    batch_size = 32
    vocab = None

    # Load and build vocab
    with open(data_path, 'r') as f:
        lines = f.read().splitlines()
    all_text = ''.join(lines)
    vocab = create_vocab(all_text)
    vocab_size = len(vocab)

    # Dataset & Loader
    train_dataset = ModData(data_path, block_size, vocab=vocab)
    val_dataset = ModData(val_path, block_size, vocab=vocab)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model configuration
    model_config = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=args.layers,
        n_head=4,
        n_embd=128,
        dropout=0.0,
        bias=True
    )
    print("Model configuration:", model_config)

    # Initialize the model
    model = GPT(model_config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.95),
        weight_decay=0.0
    )

    # Save metrics for logging
    metrics = {
        "epoch": [],
        "train_loss": [],
        "val_accuracy": [],
    }

    # Training loop
    train_iter = iter(train_loader)
    for step in range(1, args.steps + 1):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)
        logits = model(x)
        logits = logits.view(-1, vocab_size)
        y = y.view(-1)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}: loss = {loss.item():.4f}")

        # Optional: evaluate every N steps
        if step % 1000 == 0 or step == args.steps:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    logits = model(x_val)
                    preds = logits.argmax(dim=-1)
                    correct += (preds == y_val).sum().item()
                    total += y_val.numel()
            acc = correct / total
            print(f"Validation accuracy @ step {step} = {acc:.4f}")
            metrics["epoch"].append(step)  # now recording by step
            metrics["train_loss"].append(loss.item())
            metrics["val_accuracy"].append(acc)
            model.train()
    
    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    save_name = f"gpt_{args.op}_p{args.p}_{args.layers}layer_seed{args.seed}.pt"
    save_path = os.path.join("checkpoints", save_name)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Save metrics
    metrics_path = f"logs/log_{args.op}_p{args.p}_{args.layers}layer_seed{args.seed}.json"
    os.makedirs("logs", exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")