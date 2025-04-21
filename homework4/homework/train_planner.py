# I used AI on this file
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from homework.models import MLPPlanner, TransformerPlanner, CNNPlanner, save_model
from homework.metrics import PlannerMetric
from homework.datasets.road_dataset import load_data

def train(
    model_name,
    transform_pipeline=None,
    lr=1e-3,
    batch_size=128,
    num_epoch=40,
    num_workers=4,
    weight_decay=1e-4,
):
    if transform_pipeline is None:
        if model_name == "cnn_planner":
            transform_pipeline = "default"
        else:
            transform_pipeline = "state_only"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
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
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    train_loader = load_data(
        dataset_path="./drive_data/train",
        transform_pipeline=transform_pipeline,
        return_dataloader=True,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = load_data(
        dataset_path="./drive_data/val",
        transform_pipeline=transform_pipeline,
        return_dataloader=True,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False
    )
    
    best_val_loss = float('inf')
    best_metrics = None
    train_losses = []
    val_losses = []
    
    os.makedirs("./results", exist_ok=True)
    
    start_time = time.time()
    for epoch in range(num_epoch):
        model.train()
        train_loss = 0.0
        train_metric = PlannerMetric()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epoch} [Train]")
        for batch in pbar:
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            optimizer.zero_grad()
            
            if model_name == "cnn_planner":
                output = model(batch["image"])
            else:
                output = model(batch["track_left"], batch["track_right"])
                
            mask = batch["waypoints_mask"].unsqueeze(2).expand_as(batch["waypoints"])
            loss = criterion(output * mask, batch["waypoints"] * mask)
            
            train_metric.add(
                output, 
                batch["waypoints"], 
                batch["waypoints_mask"]
            )
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        train_metrics = train_metric.compute()
        
        model.eval()
        val_loss = 0.0
        val_metric = PlannerMetric()
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epoch} [Val]")
            for batch in pbar:
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                
                if model_name == "cnn_planner":
                    output = model(batch["image"])
                else:
                    output = model(batch["track_left"], batch["track_right"])
                
                mask = batch["waypoints_mask"].unsqueeze(2).expand_as(batch["waypoints"])
                loss = criterion(output * mask, batch["waypoints"] * mask)
                
                val_metric.add(
                    output, 
                    batch["waypoints"], 
                    batch["waypoints_mask"]
                )
                
                val_loss += loss.item()
                pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_metrics = val_metric.compute()
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epoch} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Train Long Err: {train_metrics['longitudinal_error']:.4f}, "
              f"Train Lat Err: {train_metrics['lateral_error']:.4f} | "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Long Err: {val_metrics['longitudinal_error']:.4f}, "
              f"Val Lat Err: {val_metrics['lateral_error']:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_metrics
            print(f"Saving best model with val_loss: {val_loss:.4f}")
            print(f"Metrics - Long Err: {val_metrics['longitudinal_error']:.4f}, "
                  f"Lat Err: {val_metrics['lateral_error']:.4f}")
            model_path = save_model(model)
            print(f"Model saved to {model_path}")
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time/60:.2f} minutes")
    print(f"Best model metrics - Long Err: {best_metrics['longitudinal_error']:.4f}, "
          f"Lat Err: {best_metrics['lateral_error']:.4f}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epoch+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epoch+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} (lr={lr}) Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./results/{model_name}_lr{lr}_loss_curve.png')
    plt.close()
    
    return model, best_val_loss, best_metrics

if __name__ == "__main__":
    for lr in [1e-2, 1e-3, 1e-4]:
        train(
            model_name="mlp_planner",
            transform_pipeline="state_only",
            num_workers=4,
            lr=lr,
            batch_size=128,
            num_epoch=40,
        )