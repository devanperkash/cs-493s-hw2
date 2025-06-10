import torch
from model import GPT, GPTConfig
from utils import create_vocab, encode, decode

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

# Initialize the model
model = GPT(model_config).to(device)
model.load_state_dict(torch.load("checkpoints/gpt_memorize.pt", map_location=device))
model.eval()

# Prepare input
prompt = "I lo"
input_ids = torch.tensor([encode(prompt, vocab)], dtype=torch.long).to(device)

# Generate tokens
max_gen_len = 23  # total length including prompt
generated = input_ids.tolist()[0]


for _ in range(max_gen_len - len(generated)):
    # crop context if needed
    idx = torch.tensor([generated[-model_config.block_size:]], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(idx)
    next_token_logits = logits[0, -1, :]
    next_token = torch.argmax(next_token_logits).item()  # greedy decoding
    generated.append(next_token)

# Decode and print
print("Input prompt:", prompt)
print("Generated:", decode(generated, vocab))