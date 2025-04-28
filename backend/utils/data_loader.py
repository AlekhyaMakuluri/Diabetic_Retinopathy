import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import logging
import numpy as np

logger = logging.getLogger(__name__)

class DRDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and labels
            split (string): 'train' or 'val'
            transform (callable, optional): Optional transform to be applied on images
        """
        self.root_dir = root_dir
        self.split = split
        # Handle both dict and callable transforms
        self.transform = transform[split] if isinstance(transform, dict) else transform
        
        # Load labels
        labels_path = os.path.join(root_dir, f'{split}_labels.csv')
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found at {labels_path}")
        
        self.labels_df = pd.read_csv(labels_path)
        logger.info(f"Loaded {len(self.labels_df)} samples for {split} split")
        
        # Verify image directories exist
        self.oct_dir = os.path.join(root_dir, 'oct', split)
        self.fundus_dir = os.path.join(root_dir, 'fundus', split)
        
        if not os.path.exists(self.oct_dir):
            raise FileNotFoundError(f"OCT directory not found at {self.oct_dir}")
        if not os.path.exists(self.fundus_dir):
            raise FileNotFoundError(f"Fundus directory not found at {self.fundus_dir}")
    
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        try:
            # Get image paths
            img_id = self.labels_df.iloc[idx]['image_id']
            oct_path = os.path.join(self.oct_dir, f'{img_id}.jpg')
            fundus_path = os.path.join(self.fundus_dir, f'{img_id}.jpg')
            
            # Load images
            oct_image = Image.open(oct_path).convert('RGB')
            fundus_image = Image.open(fundus_path).convert('RGB')
            
            # Get label and convert to class index
            label = int(self.labels_df.iloc[idx]['severity'])  # Convert to integer
            label = torch.tensor(label, dtype=torch.long)  # Convert to tensor
            
            # Apply transforms
            if self.transform:
                oct_image = self.transform(oct_image)
                fundus_image = self.transform(fundus_image)
            
            return oct_image, fundus_image, label
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {str(e)}")
            raise

class RetinopathyDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Get base directory
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Load metadata
        metadata_path = os.path.join(self.base_dir, 'data', 'processed', split, f'{split}_data.csv')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        self.metadata = pd.read_csv(metadata_path)
        logger.info(f"Loaded {len(self.metadata)} samples for {split} split")
        
        # Final normalization to be applied after augmentations
        self.final_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        try:
            row = self.metadata.iloc[idx]
            
            # Construct absolute paths
            oct_path = os.path.join(self.base_dir, row['oct_path'])
            fundus_path = os.path.join(self.base_dir, row['fundus_path'])
            
            # Load and convert to tensor
            oct_data = torch.from_numpy(np.load(oct_path)).float()
            fundus_data = torch.from_numpy(np.load(fundus_path)).float()
            label = torch.tensor(row['label'], dtype=torch.long)
            
            # Apply transforms if provided
            if self.transform:
                # Convert to PIL Image for transforms that require it
                oct_data = transforms.ToPILImage()(oct_data)
                fundus_data = transforms.ToPILImage()(fundus_data)
                
                # Apply transforms
                oct_data = self.transform(oct_data)
                fundus_data = self.transform(fundus_data)
            
            # Apply final normalization
            oct_data = self.final_transform(oct_data)
            fundus_data = self.final_transform(fundus_data)
            
            return {
                'oct': oct_data,
                'fundus': fundus_data,
                'label': label
            }
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {str(e)}")
            raise

def get_data_loaders(batch_size=32, num_workers=4):
    """Create data loaders for training, validation, and testing"""
    # Training transformations for PIL Images
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomAutocontrast(p=0.5),
        transforms.RandomEqualize(p=0.5),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor()
    ])
    
    # Validation transformations
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Create datasets
    train_dataset = RetinopathyDataset(
        data_dir='data/processed',
        split='train',
        transform=train_transform
    )
    
    val_dataset = RetinopathyDataset(
        data_dir='data/processed',
        split='val',
        transform=val_transform
    )
    
    # Create data loaders with larger batch size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def create_data_loaders(config):
    """Create data loaders for training, validation, and testing."""
    # Create datasets
    train_dataset = MultiModalDataset(
        data_dir=config.data_dir,
        labels_file=config.train_labels,
        transform=get_transforms(config.augmentation['train'])
    )
    
    val_dataset = MultiModalDataset(
        data_dir=config.data_dir,
        labels_file=config.val_labels,
        transform=get_transforms(config.augmentation['val'])
    )
    
    # Create test dataset if labels file exists
    test_dataset = None
    if os.path.exists(config.test_labels):
        test_dataset = MultiModalDataset(
            data_dir=config.data_dir,
            labels_file=config.test_labels,
            transform=get_transforms(config.augmentation['val'])  # Use validation transforms for test
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader, test_loader 