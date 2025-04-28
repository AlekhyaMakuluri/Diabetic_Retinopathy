import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import logging
from models.combined_model import CombinedModel
from train import get_val_transforms
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import wandb
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_device():
    if torch.backends.mps.is_available():
        logger.info("Using MPS")
        return torch.device("mps")
    elif torch.cuda.is_available():
        logger.info("Using CUDA")
        return torch.device("cuda")
    else:
        logger.info("Using CPU")
        return torch.device("cpu")

class RetinaDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.base_dir = os.path.dirname(csv_path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get image paths
        oct_path = os.path.join(self.base_dir, self.data.iloc[idx]['oct_path'])
        fundus_path = os.path.join(self.base_dir, self.data.iloc[idx]['fundus_path'])
        
        # Load images
        try:
            oct_image = Image.open(oct_path).convert('RGB')
            fundus_image = Image.open(fundus_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading images: {e}")
            # Create dummy images if real ones are not available
            oct_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            fundus_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        # Get label
        label = self.data.iloc[idx]['label']
        
        if self.transform:
            oct_image = self.transform(oct_image)
            fundus_image = self.transform(fundus_image)
        
        return {
            'oct': oct_image,
            'fundus': fundus_image,
            'label': torch.tensor(label, dtype=torch.long),
            'oct_path': oct_path,
            'fundus_path': fundus_path
        }

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """Plot confusion matrix"""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        logger.info("Normalized confusion matrix")
    else:
        logger.info('Confusion matrix, without normalization')

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap,
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def evaluate_modalities(model_path, batch_size=16):
    """
    Evaluate OCT and fundus modalities separately and compare their performance.
    """
    # Initialize wandb
    wandb.init(project="retinopathy-classification", job_type="modality_evaluation")
    
    # Get device
    device = get_device()
    
    # Initialize model
    model = CombinedModel()
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create a new state dict with only the encoder weights
    new_state_dict = {}
    checkpoint_state_dict = checkpoint['model_state_dict']
    
    # Copy encoder weights
    for key in model.state_dict().keys():
        if key.startswith('oct_encoder') or key.startswith('fundus_encoder'):
            if key in checkpoint_state_dict:
                new_state_dict[key] = checkpoint_state_dict[key]
            else:
                logger.warning(f"Missing key in checkpoint: {key}")
                new_state_dict[key] = model.state_dict()[key]
        else:
            new_state_dict[key] = model.state_dict()[key]
    
    # Load the filtered state dict
    model.load_state_dict(new_state_dict)
    
    # Set model to evaluation mode
    model.eval()
    
    # Get test data loader
    test_transform = get_val_transforms()
    test_dataset = RetinaDataset(
        csv_path='data/processed/test/test_data.csv',
        transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize lists to store predictions and labels
    oct_preds = []
    fundus_preds = []
    oct_probs = []
    fundus_probs = []
    true_labels = []
    
    # Evaluation loop
    with torch.no_grad():
        for batch in test_loader:
            # Move data to device
            oct_images = batch['oct'].to(device)
            fundus_images = batch['fundus'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(oct_images, fundus_images)
            
            # Get OCT predictions and probabilities
            oct_logits = outputs['logits_oct']
            oct_batch_probs = torch.softmax(oct_logits, dim=1)
            oct_batch_preds = torch.argmax(oct_batch_probs, dim=1)
            
            # Get fundus predictions and probabilities
            fundus_logits = outputs['logits_fundus']
            fundus_batch_probs = torch.softmax(fundus_logits, dim=1)
            fundus_batch_preds = torch.argmax(fundus_batch_probs, dim=1)
            
            # Store results
            oct_preds.extend(oct_batch_preds.cpu().numpy())
            fundus_preds.extend(fundus_batch_preds.cpu().numpy())
            oct_probs.extend(oct_batch_probs.cpu().numpy())
            fundus_probs.extend(fundus_batch_probs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    # Convert lists to numpy arrays
    oct_preds = np.array(oct_preds)
    fundus_preds = np.array(fundus_preds)
    oct_probs = np.array(oct_probs)
    fundus_probs = np.array(fundus_probs)
    true_labels = np.array(true_labels)
    
    # Calculate metrics for OCT
    oct_f1 = f1_score(true_labels, oct_preds, average='weighted')
    oct_auc = roc_auc_score(true_labels, oct_probs, multi_class='ovr')
    oct_report = classification_report(true_labels, oct_preds, 
                                    target_names=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR'])
    
    # Calculate metrics for fundus
    fundus_f1 = f1_score(true_labels, fundus_preds, average='weighted')
    fundus_auc = roc_auc_score(true_labels, fundus_probs, multi_class='ovr')
    fundus_report = classification_report(true_labels, fundus_preds,
                                       target_names=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR'])
    
    # Print results
    print("\nOCT Performance:")
    print(f"F1 Score: {oct_f1:.4f}")
    print(f"AUC Score: {oct_auc:.4f}")
    print("\nClassification Report:")
    print(oct_report)
    
    print("\nFundus Performance:")
    print(f"F1 Score: {fundus_f1:.4f}")
    print(f"AUC Score: {fundus_auc:.4f}")
    print("\nClassification Report:")
    print(fundus_report)
    
    # Plot confusion matrices
    from sklearn.metrics import confusion_matrix
    classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
    
    # OCT confusion matrix
    oct_cm = confusion_matrix(true_labels, oct_preds)
    plot_confusion_matrix(oct_cm, classes, title='OCT Confusion Matrix')
    
    # Fundus confusion matrix
    fundus_cm = confusion_matrix(true_labels, fundus_preds)
    plot_confusion_matrix(fundus_cm, classes, title='Fundus Confusion Matrix')
    
    # Log metrics to wandb
    wandb.log({
        "oct_f1": oct_f1,
        "oct_auc": oct_auc,
        "fundus_f1": fundus_f1,
        "fundus_auc": fundus_auc,
        "modality_comparison": wandb.Table(
            data=[
                ["OCT", oct_f1, oct_auc],
                ["Fundus", fundus_f1, fundus_auc]
            ],
            columns=["Modality", "F1 Score", "AUC Score"]
        ),
        "oct_confusion_matrix": wandb.Image('confusion_matrix.png'),
        "fundus_confusion_matrix": wandb.Image('confusion_matrix.png')
    })
    
    wandb.finish()
    
    return {
        'oct': {
            'f1': oct_f1,
            'auc': oct_auc,
            'report': oct_report
        },
        'fundus': {
            'f1': fundus_f1,
            'auc': fundus_auc,
            'report': fundus_report
        }
    }

if __name__ == "__main__":
    # Path to the saved model
    model_path = "backend/checkpoints/best_model.pth"
    evaluate_modalities(model_path) 