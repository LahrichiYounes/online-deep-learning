import argparse
import torch
import torch.utils.tensorboard as tb
from pathlib import Path
import numpy as np
from datetime import datetime
from homework.models import Classifier, save_model  # Import the model class
from homework.datasets.classification_dataset import load_data  # Import the dataset loader

# Define the training function
def train(
    exp_dir: str = "logs",
    model_name: str = "classifier",  # We'll use this to identify our model
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create a unique log directory with timestamp for storing logs and checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # Create and move the model to the appropriate device (GPU/CPU)
    model = Classifier(**kwargs)  # Make sure to pass appropriate kwargs if needed
    model = model.to(device)
    model.train()  # Set the model to training mode

    # Load the dataset for training and validation
    train_data = load_data("classification_data/train", transform_pipeline="default", batch_size=batch_size, shuffle=True)
    val_data = load_data("classification_data/val", transform_pipeline="default", batch_size=batch_size, shuffle=False)

    # Create loss function and optimizer
    loss_func = torch.nn.CrossEntropyLoss()  # Typically, cross-entropy loss for classification
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    metrics = {"train_acc": [], "val_acc": []}

    # Training loop
    for epoch in range(num_epoch):

        print("started")
        print(epoch)
        # Clear metrics at the beginning of each epoch
        for key in metrics:
            metrics[key].clear()

        model.train()  # Set model to training mode

        # Training step
        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            optimizer.zero_grad()
            logits = model(img)  # Forward pass
            loss = loss_func(logits, label)  # Compute the loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update the model parameters

            # Track accuracy
            pred = logits.argmax(1)
            metrics["train_acc"].append((pred == label).float().mean().item())
            logger.add_scalar('train/loss', loss.item(), global_step)

            global_step += 1

        # Validation step (no gradient computation)
        with torch.inference_mode():
            model.eval()  # Set the model to evaluation mode

            for img, label in val_data:
                img, label = img.to(device), label.to(device)

                logits = model(img)  # Forward pass
                pred = logits.argmax(1)
                metrics["val_acc"].append((pred == label).float().mean().item())

        # Log average train and validation accuracy
        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()

        logger.add_scalar('train/accuracy', epoch_train_acc, epoch)
        logger.add_scalar('val/accuracy', epoch_val_acc, epoch)

        # Print progress every few epochs
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
            )

    # Save the model weights
    save_model(model)

    # Save the model in the log directory for easy retrieval
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")

# Parse arguments and start training
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Arguments for training
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)  # Specify the model name
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2024)

    # Additional arguments for model hyperparameters can be added as needed
    train(**vars(parser.parse_args()))
