"""
Data loading utilities for the CUB-200-2011 dataset.
Handles image loading, part annotations, and data splits.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from . import config


def load_images_list() -> pd.DataFrame:
    """Load the list of images with their IDs."""
    images = pd.read_csv(
        config.IMAGES_FILE,
        sep=' ',
        names=['image_id', 'image_path'],
        header=None
    )
    return images


def load_train_test_split() -> pd.DataFrame:
    """Load the train/test split."""
    split = pd.read_csv(
        config.TRAIN_TEST_SPLIT_FILE,
        sep=' ',
        names=['image_id', 'is_train'],
        header=None
    )
    return split


def load_image_labels() -> pd.DataFrame:
    """Load image class labels."""
    labels = pd.read_csv(
        config.IMAGE_CLASS_LABELS_FILE,
        sep=' ',
        names=['image_id', 'class_id'],
        header=None
    )
    # Convert to 0-indexed
    labels['class_id'] = labels['class_id'] - 1
    return labels


def load_class_names() -> Dict[int, str]:
    """Load class names mapping."""
    classes = pd.read_csv(
        config.CLASSES_FILE,
        sep=' ',
        names=['class_id', 'class_name'],
        header=None
    )
    # Convert to 0-indexed
    return {row['class_id'] - 1: row['class_name'] for _, row in classes.iterrows()}


def load_bounding_boxes() -> pd.DataFrame:
    """Load bounding boxes for each image."""
    bbox = pd.read_csv(
        config.BOUNDING_BOXES_FILE,
        sep=' ',
        names=['image_id', 'x', 'y', 'width', 'height'],
        header=None
    )
    return bbox


def load_part_locations() -> pd.DataFrame:
    """Load part locations for all images."""
    parts = pd.read_csv(
        config.PART_LOCS_FILE,
        sep=' ',
        names=['image_id', 'part_id', 'x', 'y', 'visible'],
        header=None
    )
    return parts


def load_part_names() -> Dict[int, str]:
    """Load part names mapping."""
    # Part names can have spaces (e.g., "left eye"), so we parse manually
    part_names = {}
    with open(config.PARTS_FILE, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)  # Split only on first space
            if len(parts) == 2:
                part_id = int(parts[0])
                part_name = parts[1]
                part_names[part_id] = part_name
    return part_names


def get_part_mask(
    part_locations: pd.DataFrame,
    image_id: int,
    original_size: Tuple[int, int],
    target_size: int = 224,
    radius: int = 15
) -> np.ndarray:
    """
    Create a binary mask from part locations for a specific image.
    
    Args:
        part_locations: DataFrame with all part locations
        image_id: ID of the image
        original_size: Original image size (width, height)
        target_size: Target size after resizing
        radius: Radius around each part center
        
    Returns:
        Binary mask of shape (target_size, target_size)
    """
    mask = np.zeros((target_size, target_size), dtype=np.float32)
    
    # Get parts for this image
    image_parts = part_locations[part_locations['image_id'] == image_id]
    
    # Calculate scale factors
    orig_w, orig_h = original_size
    scale_x = target_size / orig_w
    scale_y = target_size / orig_h
    
    for _, part in image_parts.iterrows():
        if part['visible'] == 1:
            # Scale coordinates to target size
            x = int(part['x'] * scale_x)
            y = int(part['y'] * scale_y)
            
            # Create circular mask around the part
            for i in range(max(0, y - radius), min(target_size, y + radius + 1)):
                for j in range(max(0, x - radius), min(target_size, x + radius + 1)):
                    if (i - y) ** 2 + (j - x) ** 2 <= radius ** 2:
                        mask[i, j] = 1.0
    
    return mask


def get_part_mask_per_part(
    part_locations: pd.DataFrame,
    image_id: int,
    original_size: Tuple[int, int],
    target_size: int = 224,
    radius: int = 15
) -> Dict[int, np.ndarray]:
    """
    Create separate binary masks for each part.
    
    Returns:
        Dictionary mapping part_id to its binary mask
    """
    masks = {}
    
    image_parts = part_locations[part_locations['image_id'] == image_id]
    
    orig_w, orig_h = original_size
    scale_x = target_size / orig_w
    scale_y = target_size / orig_h
    
    for _, part in image_parts.iterrows():
        mask = np.zeros((target_size, target_size), dtype=np.float32)
        
        if part['visible'] == 1:
            x = int(part['x'] * scale_x)
            y = int(part['y'] * scale_y)
            
            for i in range(max(0, y - radius), min(target_size, y + radius + 1)):
                for j in range(max(0, x - radius), min(target_size, x + radius + 1)):
                    if (i - y) ** 2 + (j - x) ** 2 <= radius ** 2:
                        mask[i, j] = 1.0
        
        masks[part['part_id']] = mask
    
    return masks


class CUB200Dataset(Dataset):
    """
    PyTorch Dataset for CUB-200-2011.
    """
    
    def __init__(
        self,
        is_train: bool = True,
        transform: Optional[transforms.Compose] = None,
        load_parts: bool = False
    ):
        """
        Initialize the dataset.
        
        Args:
            is_train: If True, load training set; otherwise test set
            transform: Torchvision transforms to apply
            load_parts: If True, also load part annotations
        """
        self.is_train = is_train
        self.transform = transform
        self.load_parts = load_parts
        
        # Load metadata
        images = load_images_list()
        split = load_train_test_split()
        labels = load_image_labels()
        
        # Merge dataframes
        data = images.merge(split, on='image_id').merge(labels, on='image_id')
        
        # Filter by train/test
        self.data = data[data['is_train'] == (1 if is_train else 0)].reset_index(drop=True)
        
        # Load part locations if needed
        if load_parts:
            self.part_locations = load_part_locations()
        
        # Store class names
        self.class_names = load_class_names()
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        row = self.data.iloc[idx]
        
        # Load image
        image_path = config.IMAGES_DIR / row['image_path']
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (width, height)
        
        # Apply transforms
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)
        
        result = {
            'image': image_tensor,
            'label': row['class_id'],
            'image_id': row['image_id'],
            'image_path': str(image_path),
            'original_size': original_size
        }
        
        # Load part mask if needed
        if self.load_parts:
            part_mask = get_part_mask(
                self.part_locations,
                row['image_id'],
                original_size,
                target_size=config.IMAGE_SIZE,
                radius=config.PART_RADIUS
            )
            result['part_mask'] = torch.tensor(part_mask)
            
            # Also get per-part masks
            per_part_masks = get_part_mask_per_part(
                self.part_locations,
                row['image_id'],
                original_size,
                target_size=config.IMAGE_SIZE,
                radius=config.PART_RADIUS
            )
            result['per_part_masks'] = {k: torch.tensor(v) for k, v in per_part_masks.items()}
        
        return result


def get_transforms(is_train: bool = True) -> transforms.Compose:
    """Get data transforms for training or evaluation."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
        ])


def get_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
    load_parts: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """Create train and test dataloaders."""
    train_dataset = CUB200Dataset(
        is_train=True,
        transform=get_transforms(is_train=True),
        load_parts=load_parts
    )
    
    test_dataset = CUB200Dataset(
        is_train=False,
        transform=get_transforms(is_train=False),
        load_parts=load_parts
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader
