# src/model.py
import torch
import torch.nn as nn
from torchvision import models

class CUBClassifier(nn.Module):
    """Classificatore per CUB200 basato su ResNet"""
    
    def __init__(self, num_classes=200, pretrained=True, architecture='resnet50'):
        super(CUBClassifier, self).__init__()
        
        self.architecture = architecture
        
        # Carica backbone pre-addestrato
        if architecture == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_classes)
        
        elif architecture == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_classes)
        
        else:
            raise ValueError(f"Architecture {architecture} not supported")
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_features(self, x):
        """Estrai features prima del layer finale"""
        if 'resnet' in self.architecture:
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            
            return x
        
    def save(self, path):
        """Salva il modello"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'architecture': self.architecture
        }, path)
    
    @classmethod
    def load(cls, path, device='cpu'):
        """Carica il modello"""
        checkpoint = torch.load(path, map_location=device)
        model = cls(architecture=checkpoint['architecture'], pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


def get_model(pretrained=True, architecture='resnet50', device='cuda'):
    """Factory function per creare il modello"""
    model = CUBClassifier(num_classes=200, pretrained=pretrained, architecture=architecture)
    model = model.to(device)
    return model