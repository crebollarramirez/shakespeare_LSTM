from utils import *
from torch.utils.data import DataLoader
from Shakespeare_Dataset import *
from TF_LSTM import *
from LSTM import *
from train import *
import torch
import os
import glob
import sys
import argparse


def run_single_config(config_path, encoded_text, vocab_size, char_to_idx, idx_to_char):
    """Run training for a single configuration"""
    print(f"\n{'='*50}")
    print(f"Running configuration: {config_path}")
    print(f"{'='*50}")

    config = load_config(config_path)
    config["path"] = os.path.basename(config_path).replace(".yaml", "")

    seq_length = config["seq_len"]
    X, y = create_sequences(encoded_text, seq_length)

    print("CREATING SEQUENCES")
    X_tensor = torch.tensor(X, dtype=torch.long)
    y_tensor = torch.tensor(y, dtype=torch.long)

    len_data = len(y_tensor)

    train_frac, val_frac, test_frac = 0.8, 0.1, 0.1
    train_size = int(train_frac * len_data)
    val_size = int(val_frac * len_data)
    test_size = len_data - train_size - val_size

    torch.manual_seed(0)
    indices = torch.randperm(len_data)

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    assert (
        set(train_indices).isdisjoint(set(val_indices))
        and set(train_indices).isdisjoint(set(test_indices))
        and set(val_indices).isdisjoint(set(test_indices))
    )
    print("PERFORMED TRAIN/VAL/TEST SPLIT")

    X_train, y_train = X_tensor[train_indices], y_tensor[train_indices]
    X_val, y_val = X_tensor[val_indices], y_tensor[val_indices]
    X_test, y_test = X_tensor[test_indices], y_tensor[test_indices]

    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    train_dataset = ShakespeareDataset(X_train, y_train)
    train_dataloader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )

    val_dataset = ShakespeareDataset(X_val, y_val)
    val_dataloader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False
    )

    test_dataset = ShakespeareDataset(X_test, y_test)
    test_dataloader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    model = LSTMModel(
        vocab_size=vocab_size,
        embed_size=config["embed_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
    ).to(device)

    train(
        model=model,
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
    )

    ## INFERENCE
    test_loss = eval(model=model, device=device, val_dataloader=test_dataloader)
    print(f"Test loss for {config['path']}: {test_loss:.4f}")

    # Write results to file
    with open("results.txt", "a") as f:
        f.write(f"{config['path']}: {test_loss:.4f}\n")

    return test_loss


def main():
    parser = argparse.ArgumentParser(
        description="Train LSTM model with specified configuration(s)"
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="ALL",
        help='Config to run: "ALL" for all configs, or specific config filename (e.g., "base_lstm_config.yaml")',
    )

    args = parser.parse_args()

    input_file_path = "tiny_shakespeare.txt"

    print("ENCODING TEXT DATA")
    encoded_text, vocab_size, char_to_idx, idx_to_char = encode_text(input_file_path)

    # Get all config files from the configs directory
    config_files = glob.glob("configs/*.yaml")
    config_files.sort()  # Sort for consistent ordering

    if not config_files:
        print("No config files found in configs/ directory!")
        return

    # Handle argument logic
    if args.config.upper() == "ALL":
        selected_configs = config_files
        print(f"Running ALL {len(config_files)} configuration files:")
    else:
        # Check if the argument is a valid config file
        config_path = None

        # Try exact match first
        if args.config in [os.path.basename(f) for f in config_files]:
            config_path = os.path.join("configs", args.config)
        # Try with .yaml extension if not provided
        elif not args.config.endswith(".yaml"):
            candidate = args.config + ".yaml"
            if candidate in [os.path.basename(f) for f in config_files]:
                config_path = os.path.join("configs", candidate)

        if config_path and os.path.exists(config_path):
            selected_configs = [config_path]
            print(f"Running single configuration: {config_path}")
        else:
            print(f"Error: Config file '{args.config}' not found!")
            print("Available config files:")
            for config_file in config_files:
                print(f"  - {os.path.basename(config_file)}")
            return

    for config_file in selected_configs:
        print(f"  - {config_file}")

    # Clear results file at the start
    with open("results.txt", "w") as f:
        f.write("Configuration Results:\n")
        f.write("=" * 30 + "\n")

    results = {}

    # Run each configuration
    for config_file in selected_configs:
        try:
            test_loss = run_single_config(
                config_file, encoded_text, vocab_size, char_to_idx, idx_to_char
            )
            results[config_file] = test_loss
        except Exception as e:
            print(f"Error running config {config_file}: {str(e)}")
            results[config_file] = None

    # Print summary
    print(f"\n{'='*50}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*50}")

    for config_file, test_loss in results.items():
        config_name = os.path.basename(config_file)
        if test_loss is not None:
            print(f"{config_name}: {test_loss:.4f}")
        else:
            print(f"{config_name}: FAILED")


if __name__ == "__main__":
    main()
