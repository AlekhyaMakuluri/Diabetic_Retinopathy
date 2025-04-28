import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import wandb
import math
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader, TensorDataset, Dataset
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torch.utils.data import WeightedRandomSampler
from models.combined_model import CombinedModel
from utils.data_loader import get_data_loaders, RetinopathyDataset
from config import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_device():
    """Get the appropriate device for training"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device

def compute_class_weights(train_dataset):
    """Compute class weights based on training data distribution"""
    class_counts = torch.zeros(config['num_classes'])
    for batch in train_dataset:
        class_counts[batch['label']] += 1
    
    # Compute weights as inverse of frequency
    total_samples = class_counts.sum()
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights = class_weights / class_weights.sum()  # Normalize
    return class_weights

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, batch in enumerate(train_loader):
        # Get data
        oct_images = batch['oct'].to(device)
        fundus_images = batch['fundus'].to(device)
        labels = batch['label'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(oct_images, fundus_images)
        loss = criterion(outputs['logits'], labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs['logits'].data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Store predictions and labels
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Log batch progress
        if batch_idx % 10 == 0:
            logger.info(f'Batch: {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {100.*correct/total:.2f}%')
    
    # Calculate epoch metrics
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    # Calculate per-class accuracy
    class_report = classification_report(all_labels, all_preds, output_dict=True)
    logger.info(f'Training - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')
    logger.info('Per-class accuracy:')
    for class_name, metrics in class_report.items():
        if isinstance(metrics, dict):
            logger.info(f'Class {class_name}: {metrics["precision"]:.2f}, {metrics["recall"]:.2f}, {metrics["f1-score"]:.2f}')
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Get data
            oct_images = batch['oct'].to(device)
            fundus_images = batch['fundus'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(oct_images, fundus_images)
            loss = criterion(outputs['logits'], labels)
            
            # Calculate metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs['logits'].data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Log batch progress
            if batch_idx % 10 == 0:
                logger.info(f'Validation Batch: {batch_idx}/{len(val_loader)}, '
                          f'Loss: {loss.item():.4f}, '
                          f'Acc: {100.*correct/total:.2f}%')
    
    # Calculate validation metrics
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    # Calculate per-class accuracy
    class_report = classification_report(all_labels, all_preds, output_dict=True)
    logger.info(f'Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
    logger.info('Per-class accuracy:')
    for class_name, metrics in class_report.items():
        if isinstance(metrics, dict):
            logger.info(f'Class {class_name}: {metrics["precision"]:.2f}, {metrics["recall"]:.2f}, {metrics["f1-score"]:.2f}')
    
    return val_loss, val_acc

def test(model, test_loader, criterion, device):
    model.eval()
    test_preds = []
    test_labels = []
    test_probs = []
    test_loss = 0.0
    
    with torch.no_grad():
        for batch in test_loader:
            # Move data to device
            oct_data = batch['oct'].to(device)
            fundus_data = batch['fundus'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(oct_data, fundus_data)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            # Get predictions and probabilities
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            # Update metrics
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
    
    test_loss /= len(test_loader)
    test_acc = accuracy_score(test_labels, test_preds) * 100
    cm = confusion_matrix(test_labels, test_preds)
    
    return test_loss, test_acc, cm

def create_dummy_data(num_samples=100, image_size=(3, 224, 224)):
    """Create dummy data for training"""
    # Create dummy OCT images
    oct_images = torch.randn(num_samples, *image_size)
    # Create dummy fundus images
    fundus_images = torch.randn(num_samples, *image_size)
    # Create dummy labels (5 classes)
    labels = torch.randint(0, 5, (num_samples,))
    return oct_images, fundus_images, labels

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class RetinaDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Load data from the correct CSV file
        if split == 'test':
            csv_path = os.path.join(data_dir, 'test', 'test_data.csv')
        elif split == 'val':
            csv_path = os.path.join(data_dir, 'val', 'val_data.csv')
        else:  # train
            csv_path = os.path.join(data_dir, 'train', 'train_data.csv')
            
        self.data = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(self.data)} samples for {split} set")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load OCT image
        oct_path = row['oct_path']
        oct_image = np.load(oct_path)
        # Convert to correct format (H, W, C) and scale to [0, 255]
        oct_image = (oct_image * 255).astype(np.uint8)
        if oct_image.shape[0] == 1:  # If single channel
            oct_image = np.repeat(oct_image, 3, axis=0)  # Convert to RGB
        oct_image = np.transpose(oct_image, (1, 2, 0))  # CHW -> HWC
        oct_image = Image.fromarray(oct_image)
        
        # Load fundus image
        fundus_path = row['fundus_path']
        fundus_image = np.load(fundus_path)
        # Convert to correct format (H, W, C) and scale to [0, 255]
        fundus_image = (fundus_image * 255).astype(np.uint8)
        if fundus_image.shape[0] == 1:  # If single channel
            fundus_image = np.repeat(fundus_image, 3, axis=0)  # Convert to RGB
        fundus_image = np.transpose(fundus_image, (1, 2, 0))  # CHW -> HWC
        fundus_image = Image.fromarray(fundus_image)
        
        # Apply transforms if any
        if self.transform:
            oct_image = self.transform(oct_image)
            fundus_image = self.transform(fundus_image)
        
        # Get label
        label = row['label']
        
        return {
            'oct': oct_image,
            'fundus': fundus_image,
            'label': label
        }

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            oct_images = batch['oct'].to(device)
            fundus_images = batch['fundus'].to(device)
            labels = batch['label'].to(device)

            outputs = model(oct_images, fundus_images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(dataloader)
    
    # Calculate class-wise metrics
    report = classification_report(all_labels, all_preds, output_dict=True)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return accuracy, avg_loss, report, conf_matrix, all_preds, all_labels

def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_data_loaders(batch_size=32, num_workers=4):
    """Get data loaders for training and validation"""
    # Get transforms
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    
    # Create datasets
    train_dataset = RetinaDataset(
        data_dir='data/processed',
        split='train',
        transform=train_transform
    )
    
    val_dataset = RetinaDataset(
        data_dir='data/processed',
        split='val',
        transform=val_transform
    )
    
    # Create data loaders
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

def train_model(resume_from_checkpoint=None):
    # Initialize wandb
    wandb.init(project="retinopathy-classification", config={
        "learning_rate": 1e-4,
        "batch_size": 16,
        "epochs": 20,
        "weight_decay": 1e-4,
        "warmup_epochs": 5,
        "dropout_rate": 0.5
    })
    
    # Get device
    device = get_device()
    
    # Create model
    model = CombinedModel(num_classes=5).to(device)
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(
        batch_size=wandb.config.batch_size,
        num_workers=4
    )
    
    # Compute class weights
    class_weights = compute_class_weights(train_loader.dataset)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Create optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=wandb.config.learning_rate,
        weight_decay=wandb.config.weight_decay
    )
    
    # Create learning rate scheduler
    num_training_steps = len(train_loader) * wandb.config.epochs
    num_warmup_steps = len(train_loader) * wandb.config.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=10, verbose=True)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(wandb.config.epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        
        # Log metrics
        wandb.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "learning_rate": scheduler.get_last_lr()[0]
        })
        
        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            wandb.save('best_model.pth')
    
    wandb.finish()

if __name__ == '__main__':
    # Create checkpoints directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)
    
    # Check for existing checkpoint
    checkpoint_path = "checkpoints/best_model.pth" if os.path.exists("checkpoints/best_model.pth") else None
    train_model(resume_from_checkpoint=checkpoint_path) 