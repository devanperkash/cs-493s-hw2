import torch
from torch.utils.data import Dataset

def create_vocab(text):
    return sorted(set(text))

def encode(text, vocab):
    indices = []
    for char in text:
        if char in vocab:
            index = vocab.index(char)
            indices.append(index)
    return indices

def decode(indices, vocab):
    chars = []
    for i in indices:
        chars.append(vocab[i])
    return ''.join(chars)

class ModData(Dataset):
    def __init__(self, filepath, block_size, vocab=None):
        with open(filepath, 'r') as f:
            lines = f.read().splitlines()
        self.strings = lines
        self.block_size = block_size

        if vocab is None:
            all_text = ''.join(lines)
            self.vocab = create_vocab(all_text)
        else:
            self.vocab = vocab

    def __len__(self):
        return len(self.strings)
    
    def __getitem__(self, idx):
        s = self.strings[idx]
        x = encode(s[:-1], self.vocab)
        y = encode(s[1:], self.vocab)

        # Pad or truncate to block_size
        if len(x) > self.block_size:
            x = x[:self.block_size]
            y = y[:self.block_size]
        else:
            pad_len = self.block_size - len(x)
            x += [0] * pad_len
            y += [0] * pad_len

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long)
        )