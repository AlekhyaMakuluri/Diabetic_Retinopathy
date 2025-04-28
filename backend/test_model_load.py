import torch
from models.combined_model import CombinedModel
import os

def test_model_load():
    # Initialize model
    model = CombinedModel(num_classes=5)
    
    # Path to checkpoint
    checkpoint_path = os.path.join('checkpoints', 'best_model.pth')
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("Checkpoint keys:", checkpoint.keys())
        
        if 'model_state_dict' in checkpoint:
            # Try to load state dict
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("Model loaded successfully with model_state_dict")
        else:
            # Try loading the checkpoint directly
            model.load_state_dict(checkpoint, strict=False)
            print("Model loaded successfully with direct checkpoint")
            
        print("Model structure:", model)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

if __name__ == "__main__":
    test_model_load() 