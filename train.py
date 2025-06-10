import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import os
from model import GPT, GPTConfig
from utils import create_vocab, encode, decode

# Set seed for reproducibility
seed = 10
random.seed(seed)
np.random.seed(seed)
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

# Create vocabulary
context = "I love machine learning"
vocab = create_vocab(context)

# Model configuration
model_config = GPTConfig(
    block_size = len(encode(context, vocab)),
    vocab_size = len(vocab),
    n_layer = 1,
    n_head = 1,
    n_embd = 32,
    dropout = 0.0,
    bias = True
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

# Encode the input string
data = torch.tensor([encode(context, vocab)], dtype=torch.long).to(device)

# Set model to training mode
model.train()

# Training parameters
max_iters = 1000
vocab_size = len(vocab)

for step in range(max_iters):
    inputs = data
    targets = data

    # Forward pass
    logits = model(inputs)

    # Shift inputs and targets for next-token prediction
    logits = logits[:, :-1, :]  # Next token predictions
    targets = targets[:, 1:]    # True next tokens

    # Flatten for loss calculation
    logits = logits.reshape(-1, vocab_size)
    targets = targets.reshape(-1)

    # Compute per-token loss
    token_losses = F.cross_entropy(logits, targets, reduction='none')  # shape: (T-1,)

    # Create mask that ignores first 3 positions
    mask = torch.ones_like(token_losses)
    mask[:3] = 0  # ignore loss for first 3 tokens

    # Apply mask and compute the mean only on the unmasked tokens
    masked_loss = (token_losses * mask).sum() / mask.sum()

    # Backward and optimize
    optimizer.zero_grad()
    masked_loss.backward()
    optimizer.step()

    # Log progress
    if step % 100 == 0 or step == max_iters - 1:
        print(f"Step {step}: loss = {masked_loss.item():.4f}")

# Save model checkpoint
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/gpt_memorize.pt")
print("Model saved.")

# Evaluate model output
model.eval()
with torch.no_grad():
    output = model(data)
    predicted = output.argmax(dim=-1)[0].tolist()
    print("Generated:", decode(predicted, vocab))