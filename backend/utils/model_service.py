import os
import logging
from typing import Dict, Any
import torch
import wandb
from models.combined_model import CombinedModel

class ModelService:
    def __init__(self, model: CombinedModel, model_path: str = None, wandb_run_id: str = None):
        self.model = model
        self.model_path = model_path
        self.wandb_run_id = wandb_run_id
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self) -> None:
        """Load the trained model from disk or wandb."""
        try:
            if self.model_path and os.path.exists(self.model_path):
                # Load from local file
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                self.logger.info(f"Model loaded from {self.model_path}")
            elif self.wandb_run_id:
                # Load from wandb
                api = wandb.Api()
                run = api.run(f"dr-classification/{self.wandb_run_id}")
                
                # Download the model artifact
                artifact = run.logged_artifacts()[0]
                artifact_dir = artifact.download()
                model_path = os.path.join(artifact_dir, "best_model.pth")
                
                # Load the model
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                self.logger.info(f"Model loaded from wandb run {self.wandb_run_id}")
            else:
                raise FileNotFoundError("No model path or wandb run ID provided")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def predict(self, oct_image: torch.Tensor, fundus_image: torch.Tensor) -> torch.Tensor:
        """Make predictions using the loaded model."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Move images to device
            oct_image = oct_image.to(self.device)
            fundus_image = fundus_image.to(self.device)
            
            # Make prediction
            with torch.no_grad():
                logits, _ = self.model(oct_image, fundus_image)
                probs = torch.softmax(logits, dim=1)
                prediction = torch.argmax(probs, dim=1)
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            raise

    def _preprocess(self, input_data: Dict[str, Any]) -> torch.Tensor:
        """Preprocess input data for the model."""
        # Add your preprocessing logic here
        # This is a placeholder - modify according to your model's needs
        return torch.tensor(input_data["features"])

    def _postprocess(self, prediction: torch.Tensor) -> Any:
        """Postprocess model output."""
        # Add your postprocessing logic here
        # This is a placeholder - modify according to your model's needs
        return prediction.numpy().tolist() 