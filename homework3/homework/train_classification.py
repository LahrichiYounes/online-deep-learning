import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from homework.models import Classifier
from homework.datasets.classification_dataset import load_data

def train(train_path, val_path, num_epochs=10, learning_rate=0.001, batch_size=64):
    # Initialize model
    model = Classifier()
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Optional: learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Load data (with augmentation for training)
    train_loader = load_data(
        train_path, 
        transform_pipeline="aug",  # Use augmented data for training
        batch_size=batch_size, 
        shuffle=True
    )
    
    val_loader = load_data(
        val_path, 
        transform_pipeline="default",  # No augmentation for validation
        batch_size=batch_size, 
        shuffle=False
    )
    
    # Check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.inference_mode():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Statistics
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_epoch_loss)
        
        # Save best model
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            best_model_state = model.state_dict().copy()
        
        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')
        print('-' * 40)
    
    # Load best model weights
    model.load_state_dict(best_model_state)
    print(f'Best validation accuracy: {best_val_acc:.4f}')
    
    return model

# Example usage
if __name__ == "__main__":
    train_path = "path/to/train/dataset"
    val_path = "path/to/val/dataset"
    
    model = train(train_path, val_path)
    
    # Save the model
    torch.save(model.state_dict(), "classifier_model.pth")