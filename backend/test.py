import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import logging
from models.combined_model import CombinedModel
from train import RetinaDataset, get_val_transforms
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import wandb

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

def test_model(model_path, batch_size=16):
    # Initialize wandb
    wandb.init(project="retinopathy-classification", job_type="test")
    
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
        data_dir='data/processed',
        split='test',
        transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize lists to store predictions and labels
    all_preds = []
    all_preds_oct = []
    all_preds_fundus = []
    all_labels = []
    all_probs = []
    all_probs_oct = []
    all_probs_fundus = []
    
    # Test loop
    with torch.no_grad():
        for batch in test_loader:
            # Move data to device
            oct_images = batch['oct'].to(device)
            fundus_images = batch['fundus'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(oct_images, fundus_images)
            
            # Get combined predictions
            probs = torch.softmax(outputs['logits'], dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # Get OCT predictions
            probs_oct = torch.softmax(outputs['logits_oct'], dim=1)
            preds_oct = torch.argmax(probs_oct, dim=1)
            
            # Get fundus predictions
            probs_fundus = torch.softmax(outputs['logits_fundus'], dim=1)
            preds_fundus = torch.argmax(probs_fundus, dim=1)
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_preds_oct.extend(preds_oct.cpu().numpy())
            all_preds_fundus.extend(preds_fundus.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_probs_oct.extend(probs_oct.cpu().numpy())
            all_probs_fundus.extend(probs_fundus.cpu().numpy())
    
    # Calculate metrics for combined model
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    report = classification_report(all_labels, all_preds, target_names=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR'])
    cm = confusion_matrix(all_labels, all_preds)
    
    # Calculate metrics for OCT
    accuracy_oct = np.mean(np.array(all_preds_oct) == np.array(all_labels))
    report_oct = classification_report(all_labels, all_preds_oct, target_names=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR'])
    cm_oct = confusion_matrix(all_labels, all_preds_oct)
    
    # Calculate metrics for fundus
    accuracy_fundus = np.mean(np.array(all_preds_fundus) == np.array(all_labels))
    report_fundus = classification_report(all_labels, all_preds_fundus, target_names=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR'])
    cm_fundus = confusion_matrix(all_labels, all_preds_fundus)
    
    # Calculate AUC scores
    auc_score = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    auc_score_oct = roc_auc_score(all_labels, all_probs_oct, multi_class='ovr')
    auc_score_fundus = roc_auc_score(all_labels, all_probs_fundus, multi_class='ovr')
    
    # Calculate F1 scores
    f1 = f1_score(all_labels, all_preds, average='weighted')
    f1_oct = f1_score(all_labels, all_preds_oct, average='weighted')
    f1_fundus = f1_score(all_labels, all_preds_fundus, average='weighted')
    
    # Print results
    print("\nCombined Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(report)
    
    print("\nOCT Model Performance:")
    print(f"Accuracy: {accuracy_oct:.4f}")
    print(f"AUC Score: {auc_score_oct:.4f}")
    print(f"F1 Score: {f1_oct:.4f}")
    print("\nClassification Report:")
    print(report_oct)
    
    print("\nFundus Model Performance:")
    print(f"Accuracy: {accuracy_fundus:.4f}")
    print(f"AUC Score: {auc_score_fundus:.4f}")
    print(f"F1 Score: {f1_fundus:.4f}")
    print("\nClassification Report:")
    print(report_fundus)
    
    # Log metrics to wandb
    wandb.log({
        "test_accuracy": accuracy,
        "test_accuracy_oct": accuracy_oct,
        "test_accuracy_fundus": accuracy_fundus,
        "test_auc": auc_score,
        "test_auc_oct": auc_score_oct,
        "test_auc_fundus": auc_score_fundus,
        "test_f1": f1,
        "test_f1_oct": f1_oct,
        "test_f1_fundus": f1_fundus,
        "confusion_matrix": wandb.plot.confusion_matrix(
            preds=all_preds,
            y_true=all_labels,
            class_names=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
        ),
        "confusion_matrix_oct": wandb.plot.confusion_matrix(
            preds=all_preds_oct,
            y_true=all_labels,
            class_names=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
        ),
        "confusion_matrix_fundus": wandb.plot.confusion_matrix(
            preds=all_preds_fundus,
            y_true=all_labels,
            class_names=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
        )
    })
    
    wandb.finish()
    
    return {
        'combined': {
            'accuracy': accuracy,
            'auc': auc_score,
            'f1': f1,
            'report': report,
            'cm': cm
        },
        'oct': {
            'accuracy': accuracy_oct,
            'auc': auc_score_oct,
            'f1': f1_oct,
            'report': report_oct,
            'cm': cm_oct
        },
        'fundus': {
            'accuracy': accuracy_fundus,
            'auc': auc_score_fundus,
            'f1': f1_fundus,
            'report': report_fundus,
            'cm': cm_fundus
        }
    }

if __name__ == "__main__":
    # Path to the saved model
    model_path = "backend/checkpoints/best_model.pth"
    test_model(model_path) 