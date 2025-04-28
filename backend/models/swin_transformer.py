import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformer

class SwinTransformerModel(nn.Module):
    def __init__(self, num_classes=5, img_size=224, pretrained=True):
        super().__init__()
        self.swin = SwinTransformer(
            img_size=img_size,
            patch_size=4,
            in_chans=3,
            num_classes=0,  # Remove classification head
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.0,
            drop_path_rate=0.2,
            ape=False,
            patch_norm=True,
            pretrained=pretrained
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.swin.forward_features(x)
        out = self.classifier(features)
        return out

    def extract_features(self, x):
        return self.swin.forward_features(x) 