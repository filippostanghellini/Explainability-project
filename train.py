"""
Training script for the CUB-200-2011 classifier.
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from tqdm import tqdm
from typing import Dict, Tuple
import matplotlib.pyplot as plt

from src import config
from src.data_loader import get_dataloaders
from src.model import create_model, save_model


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f"{running_loss / (pbar.n + 1):.4f}",
            'acc': f"{100. * correct / total:.2f}%"
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total
    
    return test_loss, test_acc


def plot_training_curves(history: Dict, save_path: str):
    """Plot and save training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {save_path}")

#INFO: training configuration

def train(
    num_epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    model_type: str = 'resnet50',
    resume_from: str = None
):
    """Main training function."""
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Data loaders
    print("\nLoading data...")
    train_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=config.NUM_WORKERS,
        load_parts=False
    )
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Model
    print("\nCreating model...")
    model = create_model(model_type=model_type, num_classes=config.NUM_CLASSES, pretrained=True, device=device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc = 0.0
    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_acc = checkpoint.get('accuracy', 0.0)
        print(f"Resumed from epoch {start_epoch} with accuracy {best_acc:.2f}%")
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_model(
                model,
                config.MODELS_DIR / f"best_{model_type}_cub200.pth",
                optimizer=optimizer,
                epoch=epoch + 1,
                accuracy=val_acc
            )
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_model(
                model,
                config.MODELS_DIR / f"checkpoint_{model_type}_epoch{epoch + 1}.pth",
                optimizer=optimizer,
                epoch=epoch + 1,
                accuracy=val_acc
            )
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time / 60:.2f} minutes")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    
    # Plot training curves
    plot_training_curves(history, config.RESULTS_DIR / "training_curves.png")
    
    # Save final model
    save_model(
        model,
        config.MODELS_DIR / f"final_{model_type}_cub200.pth",
        optimizer=optimizer,
        epoch=num_epochs,
        accuracy=val_acc
    )
    
    return model, history

# INFO: python train.py --epochs 50 --batch_size 64 --lr 0.0005 --model resnet50

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CUB-200-2011 classifier')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50', 'vgg16'])
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        model_type=args.model,
        resume_from=args.resume
    )
