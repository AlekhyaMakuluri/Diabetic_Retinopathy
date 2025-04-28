import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

def prepare_oct_dataset(source_dir, target_dir, train_ratio=0.8):
    """Prepare OCT dataset by organizing into severity levels and splitting into train/val"""
    # Create severity level directories
    for severity in range(5):
        os.makedirs(os.path.join(target_dir, 'train', str(severity)), exist_ok=True)
        os.makedirs(os.path.join(target_dir, 'val', str(severity)), exist_ok=True)
    
    # Map original categories to severity levels
    severity_mapping = {
        'NORMAL': 0,  # No DR
        'CNV': 1,     # Mild DR
        'DME': 2,     # Moderate DR
        'DRUSEN': 3,  # Severe DR
    }
    
    # Process each category
    for category, severity in severity_mapping.items():
        category_dir = os.path.join(source_dir, category)
        if not os.path.exists(category_dir):
            continue
            
        # Get all images in category
        images = [f for f in os.listdir(category_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # Split into train and validation
        train_images, val_images = train_test_split(images, train_size=train_ratio, random_state=42)
        
        # Copy images to respective directories
        for img in train_images:
            src = os.path.join(category_dir, img)
            dst = os.path.join(target_dir, 'train', str(severity), img)
            shutil.copy2(src, dst)
            
        for img in val_images:
            src = os.path.join(category_dir, img)
            dst = os.path.join(target_dir, 'val', str(severity), img)
            shutil.copy2(src, dst)

def prepare_fundus_dataset(source_dir, target_dir, train_ratio=0.7, val_ratio=0.15):
    """Prepare fundus dataset by organizing into severity levels and splitting into train/val/test"""
    # Create severity level directories
    for severity in range(5):
        os.makedirs(os.path.join(target_dir, 'train', str(severity)), exist_ok=True)
        os.makedirs(os.path.join(target_dir, 'val', str(severity)), exist_ok=True)
        os.makedirs(os.path.join(target_dir, 'test', str(severity)), exist_ok=True)
    
    # Get all images from the gaussian filtered images directory
    images_dir = os.path.join(source_dir, 'gaussian_filtered_images', 'gaussian_filtered_images')
    if not os.path.exists(images_dir):
        raise ValueError(f"Fundus images directory not found at {images_dir}")
    
    # Map directory names to severity levels
    severity_mapping = {
        'No_DR': 0,
        'Mild': 1,
        'Moderate': 2,
        'Severe': 3,
        'Proliferate_DR': 4
    }
    
    # Process each severity level directory
    for severity_dir, severity_level in severity_mapping.items():
        severity_path = os.path.join(images_dir, severity_dir)
        if not os.path.exists(severity_path):
            continue
            
        # Get all images in severity directory
        images = [f for f in os.listdir(severity_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # First split: separate train from the rest
        train_images, temp_images = train_test_split(
            images, 
            train_size=train_ratio, 
            random_state=42
        )
        
        # Second split: separate val from test
        val_images, test_images = train_test_split(
            temp_images,
            train_size=val_ratio/(1-train_ratio),  # Adjust ratio for remaining data
            random_state=42
        )
        
        # Copy images to respective directories
        for img in train_images:
            src = os.path.join(severity_path, img)
            dst = os.path.join(target_dir, 'train', str(severity_level), img)
            shutil.copy2(src, dst)
            
        for img in val_images:
            src = os.path.join(severity_path, img)
            dst = os.path.join(target_dir, 'val', str(severity_level), img)
            shutil.copy2(src, dst)
            
        for img in test_images:
            src = os.path.join(severity_path, img)
            dst = os.path.join(target_dir, 'test', str(severity_level), img)
            shutil.copy2(src, dst)

def main():
    # Set paths relative to backend directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    oct_source = os.path.join(base_dir, 'data', 'oct', 'OCT2017 ', 'train')  # Updated path
    oct_target = os.path.join(base_dir, 'data', 'oct')
    fundus_source = os.path.join(base_dir, 'data', 'fundus')
    fundus_target = os.path.join(base_dir, 'data', 'fundus')
    
    # Prepare datasets
    print("Preparing OCT dataset...")
    prepare_oct_dataset(oct_source, oct_target)
    
    print("Preparing Fundus dataset...")
    prepare_fundus_dataset(fundus_source, fundus_target)
    
    print("Dataset preparation complete!")

if __name__ == '__main__':
    main() 