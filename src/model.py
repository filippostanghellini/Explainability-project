"""
CNN Model for CUB-200-2011 classification.
Uses a pretrained ResNet-50 with fine-tuning.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional
from . import config


class CUB200Classifier(nn.Module):
    """
    CNN classifier for CUB-200-2011 bird species classification.
    Based on pretrained ResNet-50 with modified final layer.
    """
    
    def __init__(
        self,
        num_classes: int = 200,
        pretrained: bool = True,
        freeze_backbone: bool = False #INFO: add option to freeze backbone
    ):
        """
        Initialize the classifier.
        
        Args:
            num_classes: Number of bird species classes
            pretrained: Whether to use pretrained ImageNet weights
            freeze_backbone: Whether to freeze the backbone layers
        """
        super(CUB200Classifier, self).__init__()
        
        # Load pretrained ResNet-50
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        
        # Get the number of features from the final layer
        num_features = self.backbone.fc.in_features
        
        # Replace the final fully connected layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
        
        # Optionally freeze backbone layers
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the final classification layer."""
        # Get all layers except the final fc
        modules = list(self.backbone.children())[:-1]
        feature_extractor = nn.Sequential(*modules)
        features = feature_extractor(x)
        return features.flatten(1)


def create_model(
    num_classes: int = 200,
    pretrained: bool = True,
    device: Optional[torch.device] = None,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Factory function to create the ResNet-50 model.
    
    Args:
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        device: Device to place model on
        freeze_backbone: Whether to freeze the backbone layers
        
    Returns:
        Initialized model
    """
    model = CUB200Classifier(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
    
    if device is not None:
        model = model.to(device)
    
    return model


def save_model(model: nn.Module, path: str, optimizer=None, epoch: int = 0, accuracy: float = 0.0):
    """Save model checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'accuracy': accuracy
    }
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")
    

def load_model(
    path: str,
    num_classes: int = 200,
    device: Optional[torch.device] = None
) -> nn.Module:
    """Load model from checkpoint."""
    model = create_model(num_classes=num_classes, pretrained=False)
    
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if device is not None:
        model = model.to(device)
    
    print(f"Model loaded from {path}, accuracy: {checkpoint.get('accuracy', 'N/A')}")
    return model


if __name__ == "__main__":
    # Test model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = create_model(num_classes=200, pretrained=True, device=device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224).to(device)
    output = model(x)
    print(f"Output shape: {output.shape}")
