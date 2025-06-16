from utils import *
import sys
import time
import os
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm


def train(model, device, train_dataloader, val_dataloader, config):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    patience = config.get("patience", 3)  # Default to 3 if not specified

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        pbar = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc=f"Epoch {epoch+1}/{config['epochs']}",
        )

        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            optimizer.step()
            total_loss += loss.item()

            # if batch_idx % config["log_interval"] == 0:
            # pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_dataloader)
        train_loss = eval(
            model, device, val_dataloader=train_dataloader, label="Train Evaluation"
        )
        val_loss = eval(
            model, device, val_dataloader=val_dataloader, label="Validation Evaluation"
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch [{epoch+1}/{config['epochs']}] | Average Loss: {avg_loss:.4f} | Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}"
        )

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"New best validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs without improvement"
            )
            model.load_state_dict(best_model_state)
            break

    print("Training complete!")
    fname = config["path"]
    if config["save_path"]:
        os.makedirs(config["save_path"], exist_ok=True)
        torch.save(model, os.path.join(config["save_path"], fname))
        print("Model saved")
    plot_losses(train_losses, val_losses, fname=fname)


def eval(model, device, val_dataloader, label="Evaluation"):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    with torch.no_grad():
        pbar = tqdm(val_dataloader, total=len(val_dataloader), desc=f"{label}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)  # Shape: (batch_size)

            loss = criterion(outputs, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_dataloader)
    return avg_loss
