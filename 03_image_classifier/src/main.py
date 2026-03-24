"""
Main script for training image classifier.
"""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import download_sample_images, get_data_loaders
from src.model import get_model, export_to_onnx
from src.train import Trainer, get_optimizer


def main():
    """Main training pipeline."""
    print("="*60)
    print("IMAGE CLASSIFIER - CATS VS DOGS")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    BATCH_SIZE = 32
    IMAGE_SIZE = 224
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    
    print("\n1. Preparing dataset...")
    data_dir = Path(__file__).parent.parent / 'data'
    download_sample_images(str(data_dir))
    
    train_loader, val_loader = get_data_loaders(
        str(data_dir),
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        val_split=0.2,
        num_workers=0
    )
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    
    print("\n2. Creating model...")
    print("   Training Simple CNN from scratch...")
    simple_cnn = get_model('simple_cnn', num_classes=2)
    
    optimizer = torch.optim.Adam(simple_cnn.parameters(), lr=LEARNING_RATE)
    trainer = Trainer(simple_cnn, device, optimizer=optimizer)
    
    print("\n3. Training Simple CNN...")
    history_cnn = trainer.train(
        train_loader,
        val_loader,
        num_epochs=NUM_EPOCHS // 2,
        save_dir=str(Path(__file__).parent.parent / 'models' / 'simple_cnn'),
        early_stopping_patience=3
    )
    
    print("\n4. Creating ResNet transfer learning model...")
    resnet_model = get_model('resnet18', num_classes=2, pretrained=True, freeze_backbone=True)
    
    optimizer = get_optimizer(resnet_model, lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    trainer = Trainer(resnet_model, device, optimizer=optimizer, scheduler=scheduler)
    
    print("\n5. Training ResNet (transfer learning)...")
    history_resnet = trainer.train(
        train_loader,
        val_loader,
        num_epochs=NUM_EPOCHS,
        save_dir=str(Path(__file__).parent.parent / 'models' / 'resnet'),
        early_stopping_patience=5
    )
    
    print("\n6. Exporting best model to ONNX...")
    checkpoint = torch.load(
        Path(__file__).parent.parent / 'models' / 'resnet' / 'best_model.pth',
        map_location=device
    )
    resnet_model.load_state_dict(checkpoint['model_state_dict'])
    resnet_model.eval()
    
    onnx_path = Path(__file__).parent.parent / 'models' / 'resnet' / 'model.onnx'
    export_to_onnx(resnet_model.cpu(), str(onnx_path))
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print("\nResults:")
    print(f"  Simple CNN final val accuracy: {history_cnn['val_acc'][-1]:.4f}")
    print(f"  ResNet final val accuracy: {history_resnet['val_acc'][-1]:.4f}")
    
    return resnet_model, history_resnet


def predict_image(image_path: str, model_path: str = None) -> str:
    """
    Predict class for a single image.
    
    Args:
        image_path: Path to the image
        model_path: Path to the model checkpoint
        
    Returns:
        Predicted class ('cat' or 'dog')
    """
    from PIL import Image
    from src.dataset import get_transforms
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_path is None:
        model_path = Path(__file__).parent.parent / 'models' / 'resnet' / 'best_model.pth'
    
    model = get_model('resnet18', num_classes=2, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    transform = get_transforms(train=False)
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
    
    classes = ['cat', 'dog']
    confidence = probabilities[0][predicted_class].item()
    
    return classes[predicted_class], confidence


if __name__ == "__main__":
    model, history = main()
