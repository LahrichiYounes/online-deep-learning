import argparse
import torch
import torch.utils.tensorboard as tb
from pathlib import Path
import numpy as np
from datetime import datetime
from homework.models import Detector, save_model
from homework.datasets.drive_dataset import load_data
from homework.metrics import DetectionMetric
from torch.utils.data import DataLoader

# I used AI on this file

def train(
    exp_dir: str = "logs",
    model_name: str = "detector",
    num_epoch: int = 10,
    lr: float = 1e-3,
    batch_size: int = 32,
    seg_weight: float = 1.0,
    depth_weight: float = 1.0,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = tb.SummaryWriter(log_dir)

    model = Detector(num_classes=3, **kwargs)
    model = model.to(device)
    model.train()

    def fix_array_fn(batch):
        result = {}
        for key in batch[0].keys():
            values = [item[key] for item in batch]
            
            if isinstance(values[0], np.ndarray):
                values = [arr.copy() if not arr.flags.contiguous else arr for arr in values]
                result[key] = torch.stack([torch.as_tensor(arr) for arr in values])
            else:
                result[key] = values
        return result

    train_dataset = load_data("drive_data/train", transform_pipeline="default", 
                             return_dataloader=False)
    val_dataset = load_data("drive_data/val", transform_pipeline="default", 
                           return_dataloader=False)
    
    train_data = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=fix_array_fn
    )
    
    val_data = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=fix_array_fn
    )

    segmentation_loss_fn = torch.nn.CrossEntropyLoss()
    depth_loss_fn = torch.nn.L1Loss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    metric = DetectionMetric(num_classes=3)
    best_iou = 0.0

    for epoch in range(num_epoch):
        model.train()
        train_seg_losses = []
        train_depth_losses = []
        train_total_losses = []

        for batch in train_data:
            img = batch["image"].to(device)
            seg_target = batch["track"].to(device)
            depth_target = batch["depth"].to(device)

            seg_logits, depth_pred = model(img)
            
            seg_loss = segmentation_loss_fn(seg_logits, seg_target)
            depth_loss = depth_loss_fn(depth_pred, depth_target)
            
            total_loss = seg_weight * seg_loss + depth_weight * depth_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            train_seg_losses.append(seg_loss.item())
            train_depth_losses.append(depth_loss.item())
            train_total_losses.append(total_loss.item())
            
            logger.add_scalar('train/segmentation_loss', seg_loss.item(), global_step)
            logger.add_scalar('train/depth_loss', depth_loss.item(), global_step)
            logger.add_scalar('train/total_loss', total_loss.item(), global_step)
            
            global_step += 1

        model.eval()
        metric.reset()
        val_seg_losses = []
        val_depth_losses = []
        val_total_losses = []
        
        with torch.inference_mode():
            for batch in val_data:
                img = batch["image"].to(device)
                seg_target = batch["track"].to(device)
                depth_target = batch["depth"].to(device)
                
                seg_logits, depth_pred = model(img)
                seg_pred, depth_pred = model.predict(img)
                
                seg_loss = segmentation_loss_fn(seg_logits, seg_target)
                depth_loss = depth_loss_fn(depth_pred, depth_target)
                total_loss = seg_weight * seg_loss + depth_weight * depth_loss
                
                val_seg_losses.append(seg_loss.item())
                val_depth_losses.append(depth_loss.item())
                val_total_losses.append(total_loss.item())
                
                metric.add(seg_pred, seg_target, depth_pred, depth_target)
        
        avg_train_seg_loss = np.mean(train_seg_losses)
        avg_train_depth_loss = np.mean(train_depth_losses)
        avg_train_total_loss = np.mean(train_total_losses)
        
        avg_val_seg_loss = np.mean(val_seg_losses)
        avg_val_depth_loss = np.mean(val_depth_losses)
        avg_val_total_loss = np.mean(val_total_losses)
        
        metrics_dict = metric.compute()
        val_iou = metrics_dict["iou"]
        val_accuracy = metrics_dict["accuracy"]
        val_depth_error = metrics_dict["abs_depth_error"]
        val_tp_depth_error = metrics_dict["tp_depth_error"]
        
        logger.add_scalar('val/segmentation_loss', avg_val_seg_loss, epoch)
        logger.add_scalar('val/depth_loss', avg_val_depth_loss, epoch)
        logger.add_scalar('val/total_loss', avg_val_total_loss, epoch)
        logger.add_scalar('val/iou', val_iou, epoch)
        logger.add_scalar('val/accuracy', val_accuracy, epoch)
        logger.add_scalar('val/depth_error', val_depth_error, epoch)
        logger.add_scalar('val/tp_depth_error', val_tp_depth_error, epoch)
        
        if val_iou > best_iou:
            best_iou = val_iou
            save_model(model)
            torch.save(model.state_dict(), log_dir / f"{model_name}_best.th")
            print(f"New best model saved with IoU: {val_iou:.4f}")
        
        print(
            f"Epoch {epoch + 1:2d}/{num_epoch:2d} - "
            f"Train: seg_loss={avg_train_seg_loss:.4f}, depth_loss={avg_train_depth_loss:.4f} | "
            f"Val: IoU={val_iou:.4f}, acc={val_accuracy:.4f}, depth_err={val_depth_error:.4f}, tp_depth_err={val_tp_depth_error:.4f}"
        )

    torch.save(model.state_dict(), log_dir / f"{model_name}_final.th")
    print(f"Final model saved to {log_dir / f'{model_name}_final.th'}")
    print(f"Best model (IoU={best_iou:.4f}) saved to {log_dir / f'{model_name}_best.th'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="detector")
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seg_weight", type=float, default=1.0, 
                        help="Weight for segmentation loss")
    parser.add_argument("--depth_weight", type=float, default=1.0, 
                        help="Weight for depth loss")
    parser.add_argument("--seed", type=int, default=2024)
    
    train(**vars(parser.parse_args()))