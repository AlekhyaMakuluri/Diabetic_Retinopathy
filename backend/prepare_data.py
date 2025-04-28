import os
import shutil
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreparator:
    def __init__(self, raw_data_dir, processed_data_dir):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.oct_dir = os.path.join(processed_data_dir, 'oct')
        self.fundus_dir = os.path.join(processed_data_dir, 'fundus')
        
        # Create necessary directories
        os.makedirs(self.oct_dir, exist_ok=True)
        os.makedirs(self.fundus_dir, exist_ok=True)
        
        # Image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def validate_image(self, image_path):
        """Validate if image is readable and has correct format"""
        try:
            img = Image.open(image_path)
            img.verify()
            img = Image.open(image_path).convert('RGB')
            return True
        except Exception as e:
            logger.warning(f"Invalid image {image_path}: {str(e)}")
            return False
    
    def preprocess_image(self, image_path, save_path):
        """Preprocess and save image"""
        try:
            img = Image.open(image_path).convert('RGB')
            img = self.transform(img)
            # Save as numpy array for later use
            np.save(save_path.replace('.png', '.npy'), img.numpy())
            return True
        except Exception as e:
            logger.error(f"Error preprocessing {image_path}: {str(e)}")
            return False
    
    def organize_data(self, metadata_path):
        """Organize data based on metadata file"""
        try:
            # Read metadata
            df = pd.read_csv(metadata_path)
            logger.info(f"Read metadata with {len(df)} entries")
            
            # Create data lists
            oct_paths = []
            fundus_paths = []
            labels = []
            
            for _, row in df.iterrows():
                oct_path = os.path.join(self.raw_data_dir, row['oct_path'])
                fundus_path = os.path.join(self.raw_data_dir, row['fundus_path'])
                
                if (os.path.exists(oct_path) and os.path.exists(fundus_path) and
                    self.validate_image(oct_path) and self.validate_image(fundus_path)):
                    
                    # Preprocess and save images
                    oct_save_path = os.path.join(self.oct_dir, f"{row['patient_id']}_oct.npy")
                    fundus_save_path = os.path.join(self.fundus_dir, f"{row['patient_id']}_fundus.npy")
                    
                    if (self.preprocess_image(oct_path, oct_save_path) and
                        self.preprocess_image(fundus_path, fundus_save_path)):
                        
                        oct_paths.append(oct_save_path)
                        fundus_paths.append(fundus_save_path)
                        labels.append(row['severity'])
            
            if not oct_paths:
                raise ValueError("No valid images found for preprocessing")
            
            # Split data
            train_oct, val_oct, train_fundus, val_fundus, train_labels, val_labels = train_test_split(
                oct_paths, fundus_paths, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            # Save splits
            splits = {
                'train': {'oct': train_oct, 'fundus': train_fundus, 'labels': train_labels},
                'val': {'oct': val_oct, 'fundus': val_fundus, 'labels': val_labels}
            }
            
            for split_name, split_data in splits.items():
                split_dir = os.path.join(self.processed_data_dir, split_name)
                os.makedirs(split_dir, exist_ok=True)
                
                # Save paths and labels
                pd.DataFrame({
                    'oct_path': split_data['oct'],
                    'fundus_path': split_data['fundus'],
                    'label': split_data['labels']
                }).to_csv(os.path.join(split_dir, f'{split_name}_data.csv'), index=False)
            
            logger.info("Data organization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error organizing data: {str(e)}")
            return False
    
    def create_metadata_template(self):
        """Create a template metadata CSV file"""
        template = pd.DataFrame(columns=['patient_id', 'oct_path', 'fundus_path', 'severity'])
        template.to_csv(os.path.join(self.raw_data_dir, 'metadata_template.csv'), index=False)
        logger.info("Created metadata template")

def main():
    # Set your data directories
    raw_data_dir = 'data/raw'
    processed_data_dir = 'data/processed'
    
    # Initialize data preparator
    preparator = DataPreparator(raw_data_dir, processed_data_dir)
    
    # Check if metadata exists
    metadata_path = os.path.join(raw_data_dir, 'metadata.csv')
    if not os.path.exists(metadata_path):
        preparator.create_metadata_template()
        logger.info("Please fill in the metadata template with your data information")
        return
    
    # Organize data
    logger.info("Starting data organization...")
    if preparator.organize_data(metadata_path):
        logger.info("Data preparation completed successfully")
    else:
        logger.error("Data preparation failed")

if __name__ == '__main__':
    main() 