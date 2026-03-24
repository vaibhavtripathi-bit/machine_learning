"""
Model module for image classification.
Implements custom CNN and transfer learning with ResNet.
"""

from typing import Optional
import torch
import torch.nn as nn
import torchvision.models as models


class SimpleCNN(nn.Module):
    """Simple CNN for image classification from scratch."""
    
    def __init__(self, num_classes: int = 2):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNetTransfer(nn.Module):
    """Transfer learning model using pretrained ResNet."""
    
    def __init__(
        self,
        num_classes: int = 2,
        model_name: str = 'resnet18',
        pretrained: bool = True,
        freeze_backbone: bool = True
    ):
        """
        Initialize the transfer learning model.
        
        Args:
            num_classes: Number of output classes
            model_name: ResNet variant ('resnet18', 'resnet34', 'resnet50')
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze the backbone layers
        """
        super(ResNetTransfer, self).__init__()
        
        if model_name == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
        elif model_name == 'resnet34':
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet34(weights=weights)
        elif model_name == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def unfreeze_backbone(self, num_layers: Optional[int] = None):
        """
        Unfreeze backbone layers for fine-tuning.
        
        Args:
            num_layers: Number of layers to unfreeze from the end.
                       If None, unfreeze all layers.
        """
        if num_layers is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            layers = list(self.backbone.children())[:-1]
            for layer in layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True


def get_model(
    model_type: str = 'resnet18',
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True
) -> nn.Module:
    """
    Get a model for image classification.
    
    Args:
        model_type: Type of model ('simple_cnn', 'resnet18', 'resnet34', 'resnet50')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights (for ResNet)
        freeze_backbone: Whether to freeze backbone (for ResNet)
        
    Returns:
        PyTorch model
    """
    if model_type == 'simple_cnn':
        return SimpleCNN(num_classes)
    elif model_type in ['resnet18', 'resnet34', 'resnet50']:
        return ResNetTransfer(num_classes, model_type, pretrained, freeze_backbone)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def export_to_onnx(
    model: nn.Module,
    save_path: str,
    input_size: tuple = (1, 3, 224, 224)
) -> None:
    """
    Export model to ONNX format.
    
    Args:
        model: PyTorch model
        save_path: Path to save the ONNX model
        input_size: Input tensor size
    """
    model.eval()
    dummy_input = torch.randn(input_size)
    
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model exported to ONNX: {save_path}")
