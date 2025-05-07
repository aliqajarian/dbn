import torch
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple
import pickle
from google.colab import drive
from models.dbn import TimeSeriesDBN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, drive_path: str = "/content/drive/MyDrive/anomaly_detection"):
        self.drive_path = Path(drive_path)
        self.model_dir = self.drive_path / "models"
    
    def load_model(self, model_name: str) -> Tuple[TimeSeriesDBN, Dict]:
        """Load model and metadata from Google Drive."""
        model_path = self.model_dir / f"{model_name}.pth"
        metadata_path = self.model_dir / f"{model_name}_metadata.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model
        model = TimeSeriesDBN(
            layer_sizes=[3, 64, 32, 16],
            sequence_length=10
        )
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # Load metadata
        metadata = None
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
        
        logger.info(f"Model loaded from {model_path}")
        return model, metadata
    
    def predict(self, model: TimeSeriesDBN, data: pd.DataFrame) -> pd.DataFrame:
        """Make predictions using the loaded model."""
        # Prepare features
        features = self.prepare_features(data)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features)
        
        # Get predictions
        with torch.no_grad():
            reconstruction = model.reconstruct(features_tensor)
            errors = torch.mean((features_tensor - reconstruction) ** 2, dim=1)
        
        # Add predictions to dataframe
        data['reconstruction_error'] = errors.numpy()
        data['is_anomaly'] = errors.numpy() > errors.mean() + 2 * errors.std()
        
        return data
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for prediction."""
        # Normalize numerical features
        features = df[['price', 'rating', 'review_count']].values
        features = (features - features.mean(axis=0)) / features.std(axis=0)
        return features

def main():
    # Mount Google Drive
    drive.mount('/content/drive')
    
    # Initialize model loader
    loader = ModelLoader()
    
    try:
        # Load model
        model, metadata = loader.load_model("amazon_anomaly_detector")
        
        # Print model metadata
        if metadata:
            logger.info("\nModel Metadata:")
            for key, value in metadata.items():
                logger.info(f"{key}: {value}")
        
        # Example: Load some new data for prediction
        # df = pd.read_csv("new_data.csv")
        # results = loader.predict(model, df)
        # results.to_csv("predictions.csv", index=False)
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")

if __name__ == "__main__":
    main() 