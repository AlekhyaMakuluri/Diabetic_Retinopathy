import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .swin_transformer import SwinTransformerModel
from .gnn import GNNModel
from .ssl_module import SSLModule

class CombinedModel(nn.Module):
    def __init__(self, num_classes=5):
        super(CombinedModel, self).__init__()
        
        # Load pretrained ResNet50 for both OCT and fundus images
        self.oct_encoder = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.fundus_encoder = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Remove the last fully connected layer
        self.oct_encoder.fc = nn.Identity()
        self.fundus_encoder.fc = nn.Identity()
        
        # Feature projection layers with consistent dimensions
        self.oct_projection = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.fundus_projection = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Cross-attention mechanism
        self.oct_attention = nn.MultiheadAttention(512, num_heads=8)
        self.fundus_attention = nn.MultiheadAttention(512, num_heads=8)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, oct_images, fundus_images):
        # Extract features from both encoders
        oct_features = self.oct_encoder(oct_images)
        fundus_features = self.fundus_encoder(fundus_images)
        
        # Project features to consistent dimension
        oct_proj = self.oct_projection(oct_features)
        fundus_proj = self.fundus_projection(fundus_features)
        
        # Apply cross-attention
        oct_attn, _ = self.oct_attention(oct_proj.unsqueeze(0), fundus_proj.unsqueeze(0), fundus_proj.unsqueeze(0))
        fundus_attn, _ = self.fundus_attention(fundus_proj.unsqueeze(0), oct_proj.unsqueeze(0), oct_proj.unsqueeze(0))
        
        # Remove extra dimension from attention output
        oct_attn = oct_attn.squeeze(0)
        fundus_attn = fundus_attn.squeeze(0)
        
        # Concatenate attended features
        combined = torch.cat([oct_attn, fundus_attn], dim=1)
        
        # Fuse features
        fused = self.fusion(combined)
        
        # Get final predictions
        logits = self.classifier(fused)
        
        return {
            'logits': logits,
            'oct_features': oct_proj,
            'fundus_features': fundus_proj,
            'fused_features': fused
        } 