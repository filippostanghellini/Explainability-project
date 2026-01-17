# src/data_utils.py
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CUBDataset(Dataset):
    """Dataset class per CUB200-2011"""
    
    def __init__(self, dataset_path, image_ids=None, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        
        # Carica metadati
        self.images_df = pd.read_csv(
            os.path.join(dataset_path, 'images.txt'),
            sep=' ', names=['img_id', 'filepath']
        )
        self.labels_df = pd.read_csv(
            os.path.join(dataset_path, 'image_class_labels.txt'),
            sep=' ', names=['img_id', 'class_id']
        )
        
        # Merge
        self.data = self.images_df.merge(self.labels_df, on='img_id')
        
        # Filtra se necessario
        if image_ids is not None:
            self.data = self.data[self.data['img_id'].isin(image_ids)]
        
        self.data = self.data.reset_index(drop=True)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Carica immagine
        img_path = os.path.join(self.dataset_path, 'images', row['filepath'])
        image = Image.open(img_path).convert('RGB')
        
        # Applica trasformazioni
        if self.transform:
            image = self.transform(image)
        
        label = row['class_id'] - 1  # classi da 0 a 199
        
        return {
            'image': image,
            'label': label,
            'img_id': row['img_id']
        }

def get_part_annotations(dataset_path, img_id):
    """Ottieni annotazioni delle parti per un'immagine"""
    parts_df = pd.read_csv(
        os.path.join(dataset_path, 'parts/part_locs.txt'),
        sep=' ',
        names=['img_id', 'part_id', 'x', 'y', 'visible']
    )
    
    img_parts = parts_df[parts_df['img_id'] == img_id]
    return img_parts[img_parts['visible'] == 1]  # solo parti visibili