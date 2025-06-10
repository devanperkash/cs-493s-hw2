import random
import os

def compute_mod_result(a, b, p, operation):
    if operation == "+":
        return (a + b) % p
    elif operation == "-":
        return (a - b) % p
    elif operation == "/":
        if b == 0:
            return None  # Cannot divide by zero
        try:
            b_inv = pow(b, -1, p)  # Modular inverse of b mod p
            return (a * b_inv) % p
        except ValueError:
            return None  # No modular inverse exists (shouldn't happen if p is prime)
    else:
        return None
    
def generate_dataset(p, operation):
    examples = []
    for a in range(p+1):
        for b in range(p+1):
            c = compute_mod_result(a, b, p, operation)
            if c is not None:
                example = f"{a} {operation} {b} = {c}"
                examples.append(example)
    return examples

def split_dataset(data, seed=10):
    random.seed(seed)
    random.shuffle(data)
    length = len(data)
    train = data[:int(length * 0.8)]
    val = data[int(length * 0.8):int(length * 0.9)]
    test = data[int(length * 0.9):]
    return train, val, test

def save_split(split, out_path):
    with open(out_path, 'w') as f:
        f.write('\n'.join(split))

def generate_and_save_all_data():
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    moduli = [97, 113]
    operations = ['+', '-', '/']
    op_symbol_map = {'+': 'plus', '-': 'minus', '/': 'div'}

    for p in moduli:
        for op in operations:
            print(f"Generating data for '{op}' mod {p}...")
            data = generate_dataset(p, op)
            train, val, test = split_dataset(data)

            op_str = op_symbol_map[op]  # Convert to filename-safe string

            save_split(train, f"{output_dir}/{op_str}_p{p}_train.txt")
            save_split(val,   f"{output_dir}/{op_str}_p{p}_val.txt")
            save_split(test,  f"{output_dir}/{op_str}_p{p}_test.txt")

            print(f"  ➤ {len(train)} train, {len(val)} val, {len(test)} test samples")

if __name__ == "__main__":
    generate_and_save_all_data()