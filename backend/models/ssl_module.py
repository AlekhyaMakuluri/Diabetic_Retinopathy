import torch
import torch.nn as nn
import torch.nn.functional as F

class SSLModule(nn.Module):
    def __init__(self, feature_dim=256, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
    
    def forward(self, x1, x2):
        # Project features
        z1 = self.projection(x1)  # Shape: (B, 128)
        z2 = self.projection(x2)  # Shape: (B, 128)
        
        # Normalize features
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Concatenate all features
        features = torch.cat([z1, z2], dim=0)  # Shape: (2B, 128)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.mT)  # Shape: (2B, 2B)
        
        # Create labels for positive pairs
        batch_size = z1.shape[0]
        labels = torch.arange(batch_size, device=z1.device)
        labels = torch.cat([labels, labels])  # Shape: (2B,)
        
        # Create mask for positive pairs
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # Shape: (2B, 2B)
        mask = mask.fill_diagonal_(0)  # Remove self-similarity
        
        # Compute logits
        logits = similarity_matrix / self.temperature  # Shape: (2B, 2B)
        
        # Compute loss
        exp_logits = torch.exp(logits)  # Shape: (2B, 2B)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # Shape: (2B, 2B)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)  # Shape: (2B,)
        loss = -mean_log_prob_pos.mean()  # Shape: scalar
        
        return loss 