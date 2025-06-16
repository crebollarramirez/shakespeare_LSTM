import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import yaml
from tqdm import tqdm


def encode_text(input_file_path):
    # lets load the file
    with open(input_file_path, "r", encoding="utf-8") as f:
        text = f.read()

        # create character encoding
        chars = sorted(set(text))
        vocab_size = len(chars)
        char_to_idx = {ch: i for i, ch in enumerate(chars)}
        idx_to_char = {i: ch for i, ch in enumerate(chars)}

        # Convert text to numerical format with progress bar
        encoded_text = [
            char_to_idx[c] for c in tqdm(text, desc="Encoding text", unit="char")
        ]

        return encoded_text, vocab_size, char_to_idx, idx_to_char


def create_sequences(data, seq_length):
    # Convert to numpy array first if it's a list
    if isinstance(data, list):
        data = np.array(data, dtype=np.int32)

    num_sequences = len(data) - seq_length

    if num_sequences <= 0:
        raise ValueError(f"Data too short for seq_length {seq_length}")

    print(f"Creating {num_sequences:,} sequences of length {seq_length}")

    # Ultra-fast approach using numpy stride tricks
    from numpy.lib.stride_tricks import sliding_window_view

    # Create sliding windows
    windows = sliding_window_view(data, window_shape=seq_length + 1)

    X = windows[:num_sequences, :-1].copy()  # All but last character
    y = windows[:num_sequences, -1].copy()  # Last character

    return X, y


def plot_losses(train_losses, val_losses, fname):
    # Create 'plots' directory if it doesn't exist

    if not os.path.isdir("plots"):
        os.mkdir("plots")

    # Plotting training and validation losses
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch")

    # Saving the plot as an image file in 'plots' directory
    plt.savefig("./plots/" + fname + ".png")


# Text generation function
def generate_text(model, device, char_idx_map, idx_to_char, max_len=1000, temp=0.8):
    model.eval()
    start_text = "I walked into the "
    input_seq = [char_idx_map[c] for c in start_text]
    input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)

    model.to(device)
    generated_text = start_text
    for _ in range(max_len):
        with torch.no_grad():
            output = model(input_tensor)
            output = output.squeeze().div(temp).exp()
            predicted_idx = torch.multinomial(output, 1).item()
            predicted_char = idx_to_char[predicted_idx]
            generated_text += predicted_char
            input_tensor = torch.cat(
                (
                    input_tensor[:, 1:],
                    torch.tensor([[predicted_idx]], dtype=torch.long).to(device),
                ),
                dim=1,
            )

    return generated_text


def load_config(file_path):

    with open(file_path, "r") as file:
        config = yaml.safe_load(file)

    return config
