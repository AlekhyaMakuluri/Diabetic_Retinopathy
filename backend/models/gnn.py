import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, SAGEConv, GATConv
from torch_geometric.data import Data

class GNNModel(nn.Module):
    def __init__(self, hidden_channels=1024):
        super().__init__()
        
        # Image feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Graph construction and processing
        self.gcn1 = GCNConv(256, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.gcn3 = GCNConv(hidden_channels, hidden_channels)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_channels, num_heads=8)
        
        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x):
        # Extract image features
        features = self.feature_extractor(x)
        batch_size = features.size(0)
        
        # Reshape features for graph processing
        features = features.view(batch_size, -1, 256)  # [batch_size, num_nodes, 256]
        
        # Create fully connected graph
        edge_index = self._create_fully_connected_graph(features.size(1))
        edge_index = edge_index.to(features.device)
        
        # Process through GCN layers
        x = self.gcn1(features, edge_index)
        x = F.relu(x)
        x = self.gcn2(x, edge_index)
        x = F.relu(x)
        x = self.gcn3(x, edge_index)
        
        # Apply attention
        x = x.transpose(0, 1)  # [num_nodes, batch_size, hidden_channels]
        x, _ = self.attention(x, x, x)
        x = x.transpose(0, 1)  # [batch_size, num_nodes, hidden_channels]
        
        # Global average pooling
        x = x.mean(dim=1)  # [batch_size, hidden_channels]
        
        # Final projection
        x = self.projection(x)
        
        return x
    
    def _create_fully_connected_graph(self, num_nodes):
        # Create a fully connected graph
        edge_index = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index.append([i, j])
        return torch.tensor(edge_index, dtype=torch.long).t()

    def create_graph_from_features(self, features, k=12):
        """Create a graph from feature vectors using k-nearest neighbors"""
        # Calculate pairwise distances
        dist = torch.cdist(features, features)
        
        # Get k-nearest neighbors for each node
        _, indices = torch.topk(dist, k=k, dim=1, largest=False)
        
        # Create edge index
        edge_index = []
        for i in range(len(features)):
            for j in indices[i]:
                edge_index.append([i, j])
                edge_index.append([j, i])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return edge_index 