#!/usr/bin/env python3
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Import the required modules from homework
from homework.models import MLPPlanner, TransformerPlanner, CNNPlanner, save_model
from homework.metrics import longitudinal_error, lateral_error
from homework.datasets.road_dataset import RoadDataset

def train(
    model_name,
    transform_pipeline,
    output_dir="./results",
    num_workers=4,
    lr=1e-3,
    batch_size=128,
    epochs=40,
    weight_decay=1e-4,
    device=None,
):
    """
    Train a planner model and save the best model based on validation loss
    
    Args:
        model_name (str): One of "mlp_planner", "transformer_planner", "cnn_planner"
        transform_pipeline (str): Type of transformation to apply to the data
        output_dir (str): Directory to save results
        num_workers (int): Number of workers for data loading
        lr (float): Learning rate
        batch_size (int): Batch size
        epochs (int): Number of epochs to train
        weight_decay (float): Weight decay for optimizer
        device (torch.device): Device to use for training (defaults to cuda if available)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model based on model_name
    if model_name == "mlp_planner":
        model = MLPPlanner()
    elif model_name == "transformer_planner":
        model = TransformerPlanner()
    elif model_name == "cnn_planner":
        model = CNNPlanner()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    model = model.to(device)
    print(f"Initialized {model_name}")
    
    # Define loss function
    criterion = nn.MSELoss()
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Load datasets
    train_dataset = RoadDataset(split="train", transform_pipeline=transform_pipeline)
    val_dataset = RoadDataset(split="val", transform_pipeline=transform_pipeline)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    print(f"Training with {len(train_dataset)} samples, validating with {len(val_dataset)} samples")
    
    # Training metrics tracking
    best_val_loss = float('inf')
    best_long_error = float('inf')
    best_lat_error = float('inf')
    train_losses = []
    val_losses = []
    train_long_errors = []
    train_lat_errors = []
    val_long_errors = []
    val_lat_errors = []
    
    # Training loop
    start_time = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # ===== Training phase =====
        model.train()
        train_loss = 0.0
        train_long_error = 0.0
        train_lat_error = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            if model_name == "cnn_planner":
                output = model(batch["image"])
            else:
                output = model(batch["track_left"], batch["track_right"])
                
            # Compute loss using mask
            mask = batch["waypoints_mask"].unsqueeze(2).expand_as(batch["waypoints"])
            pred_waypoints = output * mask
            target_waypoints = batch["waypoints"] * mask
            
            loss = criterion(pred_waypoints, target_waypoints)
            
            # Compute metrics for monitoring
            with torch.no_grad():
                long_err = longitudinal_error(pred_waypoints, target_waypoints, mask)
                lat_err = lateral_error(pred_waypoints, target_waypoints, mask)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_long_error += long_err.item()
            train_lat_error += lat_err.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'long_err': f"{long_err.item():.4f}",
                'lat_err': f"{lat_err.item():.4f}"
            })
        
        # Calculate average training metrics
        train_loss /= len(train_loader)
        train_long_error /= len(train_loader)
        train_lat_error /= len(train_loader)
        
        train_losses.append(train_loss)
        train_long_errors.append(train_long_error)
        train_lat_errors.append(train_lat_error)
        
        # ===== Validation phase =====
        model.eval()
        val_loss = 0.0
        val_long_error = 0.0
        val_lat_error = 0.0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for batch in pbar:
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                
                # Forward pass
                if model_name == "cnn_planner":
                    output = model(batch["image"])
                else:
                    output = model(batch["track_left"], batch["track_right"])
                
                # Compute loss using mask
                mask = batch["waypoints_mask"].unsqueeze(2).expand_as(batch["waypoints"])
                pred_waypoints = output * mask
                target_waypoints = batch["waypoints"] * mask
                
                loss = criterion(pred_waypoints, target_waypoints)
                
                # Compute metrics
                long_err = longitudinal_error(pred_waypoints, target_waypoints, mask)
                lat_err = lateral_error(pred_waypoints, target_waypoints, mask)
                
                # Update metrics
                val_loss += loss.item()
                val_long_error += long_err.item()
                val_lat_error += lat_err.item()
                
                pbar.set_postfix({
                    'val_loss': f"{loss.item():.4f}",
                    'long_err': f"{long_err.item():.4f}",
                    'lat_err': f"{lat_err.item():.4f}"
                })
        
        # Calculate average validation metrics
        val_loss /= len(val_loader)
        val_long_error /= len(val_loader)
        val_lat_error /= len(val_loader)
        
        val_losses.append(val_loss)
        val_long_errors.append(val_long_error)
        val_lat_errors.append(val_lat_error)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s - "
              f"Train Loss: {train_loss:.4f}, "
              f"Train Long Err: {train_long_error:.4f}, "
              f"Train Lat Err: {train_lat_error:.4f} | "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Long Err: {val_long_error:.4f}, "
              f"Val Lat Err: {val_lat_error:.4f}")
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_long_error = val_long_error
            best_lat_error = val_lat_error
            print(f"Saving best model with val_loss: {val_loss:.4f}, long_err: {val_long_error:.4f}, lat_err: {val_lat_error:.4f}")
            model_path = save_model(model)
            print(f"Model saved to {model_path}")
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.2f} minutes!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation metrics - Long Error: {best_long_error:.4f}, Lat Error: {best_lat_error:.4f}")
    
    # Plot training curves
    plot_training_curves(
        train_losses, val_losses, 
        train_long_errors, val_long_errors,
        train_lat_errors, val_lat_errors,
        model_name, output_dir
    )
    
    return model, best_val_loss, best_long_error, best_lat_error

def plot_training_curves(train_losses, val_losses, train_long_errs, val_long_errs, 
                         train_lat_errs, val_lat_errs, model_name, output_dir):
    """Plot and save the training curves"""
    epochs = range(1, len(train_losses) + 1)
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot loss
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot longitudinal error
    ax2.plot(epochs, train_long_errs, 'b-', label='Train Long Error')
    ax2.plot(epochs, val_long_errs, 'r-', label='Val Long Error')
    ax2.set_title('Longitudinal Error')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Error')
    ax2.legend()
    ax2.grid(True)
    
    # Plot lateral error
    ax3.plot(epochs, train_lat_errs, 'b-', label='Train Lat Error')
    ax3.plot(epochs, val_lat_errs, 'r-', label='Val Lat Error')
    ax3.set_title('Lateral Error')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Error')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    
    # Save the figure
    fig_path = os.path.join(output_dir, f'{model_name}_training_curves.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"Training curves saved to {fig_path}")

def main():
    parser = argparse.ArgumentParser(description="Train a planner model")
    parser.add_argument('--model', type=str, required=True, 
                        choices=['mlp_planner', 'transformer_planner', 'cnn_planner'],
                        help='Model type to train')
    parser.add_argument('--transform', type=str, default=None,
                        help='Transform pipeline to use (defaults based on model)')
    parser.add_argument('--lr', type=float, default=1e-3, 
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=40,
                        help='Number of epochs to train')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Set default transform based on model if not specified
    if args.transform is None:
        if args.model == 'cnn_planner':
            args.transform = 'image_only'
        else:
            args.transform = 'state_only'
    
    print(f"Training {args.model} with {args.transform} transform...")
    
    # Train the model
    train(
        model_name=args.model,
        transform_pipeline=args.transform,
        output_dir=args.output_dir,
        num_workers=args.workers,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs
    )

if __name__ == "__main__":
    main()