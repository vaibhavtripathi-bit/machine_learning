"""
Dataset module for image classification.
Handles data loading, transforms, and DataLoader creation.
"""

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets
from PIL import Image


class CatsDogsDataset(Dataset):
    """Custom dataset for Cats vs Dogs classification."""
    
    def __init__(self, root_dir: str, transform: Optional[transforms.Compose] = None):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory containing 'cats' and 'dogs' folders
            transform: Torchvision transforms to apply
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        self.images = []
        self.labels = []
        
        for label, class_name in enumerate(['cats', 'dogs']):
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    self.images.append(str(img_path))
                    self.labels.append(label)
                for img_path in class_dir.glob('*.png'):
                    self.images.append(str(img_path))
                    self.labels.append(label)
        
        self.classes = ['cat', 'dog']
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='gray')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(image_size: int = 224, train: bool = True) -> transforms.Compose:
    """
    Get transforms for training or evaluation.
    
    Args:
        image_size: Target image size
        train: Whether to include data augmentation
        
    Returns:
        Composed transforms
    """
    if train:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def create_sample_dataset(data_dir: str, num_samples: int = 100) -> str:
    """
    Create a sample dataset with synthetic images for testing.
    
    Args:
        data_dir: Directory to create the dataset
        num_samples: Number of samples per class
        
    Returns:
        Path to the created dataset
    """
    import numpy as np
    
    data_path = Path(data_dir)
    
    for class_name in ['cats', 'dogs']:
        class_dir = data_path / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(num_samples):
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            if class_name == 'cats':
                img_array[:, :, 0] = np.clip(img_array[:, :, 0] + 50, 0, 255)
            else:
                img_array[:, :, 2] = np.clip(img_array[:, :, 2] + 50, 0, 255)
            
            img = Image.fromarray(img_array)
            img.save(class_dir / f'{class_name}_{i:04d}.jpg')
    
    print(f"Created sample dataset with {num_samples} images per class at {data_path}")
    return str(data_path)


def get_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    val_split: float = 0.2,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        data_dir: Root directory of the dataset
        batch_size: Batch size
        image_size: Target image size
        val_split: Validation set fraction
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_transform = get_transforms(image_size, train=True)
    val_transform = get_transforms(image_size, train=False)
    
    full_dataset = CatsDogsDataset(data_dir, transform=train_transform)
    
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    val_dataset.dataset.transform = val_transform
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def download_sample_images(data_dir: str) -> str:
    """
    Download sample images or create synthetic ones for demonstration.
    
    Args:
        data_dir: Directory to save images
        
    Returns:
        Path to the dataset
    """
    data_path = Path(data_dir)
    
    if (data_path / 'cats').exists() and (data_path / 'dogs').exists():
        cat_count = len(list((data_path / 'cats').glob('*.jpg')))
        dog_count = len(list((data_path / 'dogs').glob('*.jpg')))
        if cat_count > 0 and dog_count > 0:
            print(f"Dataset already exists: {cat_count} cats, {dog_count} dogs")
            return str(data_path)
    
    print("Creating synthetic dataset for demonstration...")
    return create_sample_dataset(str(data_path), num_samples=200)
