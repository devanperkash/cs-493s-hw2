import torch
import torch.nn.functional as F
from model import GPT, GPTConfig
from utils import create_vocab, encode, decode, ModData
from torch.utils.data import DataLoader
from torch.optim import AdamW
import os
import random
import argparse
import json
from datetime import datetime

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--op', type=str, default='/', choices=['+', '-', '/'])
parser.add_argument('--p', type=int, default=97, choices=[97, 113])
parser.add_argument('--layers', type=int, default=1, choices=[1, 2])
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--train_frac', type=float, default=0.005)
args = parser.parse_args()

# Create checkpoint directory
op_name = {'+': 'add', '-': 'sub', '/': 'div'}[args.op]
ckpt_dir = f"checkpoints/grok/{op_name}_p{args.p}_{args.layers}layer_seed{args.seed}"
os.makedirs(ckpt_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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
    vocab = None

    # Load and build vocab
    with open(data_path, 'r') as f:
        lines = f.read().splitlines()
    all_text = ''.join(lines)
    vocab = create_vocab(all_text)
    vocab_size = len(vocab)

    # Dataset & Loader
    full_train_dataset = ModData(data_path, block_size, vocab=vocab)
    train_size = int(len(full_train_dataset) * args.train_frac)
    generator = torch.Generator().manual_seed(seed)
    train_dataset, _ = torch.utils.data.random_split(full_train_dataset, [train_size, len(full_train_dataset) - train_size], generator=generator)

    val_dataset = ModData(val_path, block_size, vocab=vocab)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

    # Model configuration
    model_config = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=args.layers,
        n_head=1,
        n_embd=16,
        dropout=0.0,
        bias=True
    )
    print("Model configuration:", model_config)

    # Initialize the model
    model = GPT(model_config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-2
    )

    # Save metrics for logging
    metrics = {
        "epoch": [],
        "train_loss": [],
        "val_accuracy": [],
        "val_loss": [],
    }

    # Training loop
    epochs = args.epochs
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            logits = model(x)

            logits = logits.view(-1, vocab_size)
            y = y.view(-1)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step % 100 == 0:
                print(f"Epoch {epoch+1} Step {step}: loss = {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished: avg train loss = {avg_loss:.4f}")

        if (epoch + 1) % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f"epoch{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[Checkpoint] Saved model at epoch {epoch+1} to {ckpt_path}")
            
            intermediate_metrics_path = os.path.join("logs", f"grok_{op_name}_p{args.p}_{args.layers}layer_seed{args.seed}_{timestamp}_epoch{epoch+1}.json")
            with open(intermediate_metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

        # Validation accuracy
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.numel()
        acc = correct / total
        print(f"           Validation accuracy = {acc:.4f}")

        # Validation loss
        val_total_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                val_total_loss += loss.item()
        avg_val_loss = val_total_loss / len(val_loader)
        metrics["val_loss"].append(avg_val_loss)

        metrics["epoch"].append(epoch + 1)
        metrics["train_loss"].append(avg_loss)
        metrics["val_accuracy"].append(acc)

    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    save_name = f"gpt_grok_{op_name}_p{args.p}_{args.layers}layer_seed{args.seed}_{timestamp}.pt"
    save_path = os.path.join("checkpoints", save_name)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Save metrics
    metrics_path = f"logs/grok_{op_name}_p{args.p}_{args.layers}layer_seed{args.seed}_{timestamp}.json"
    os.makedirs("logs", exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")