import torch
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple
import pickle
from google.colab import drive
from models.dbn import TimeSeriesDBN
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ColabCheckpoint:
    def __init__(self, drive_path: str = "/content/drive/MyDrive/anomaly_detection"):
        """Initialize checkpoint manager for Google Colab."""
        self.drive_path = Path(drive_path)
        self.checkpoint_dir = self.drive_path / "checkpoints"
        self.data_dir = self.drive_path / "data"
        self.model_dir = self.drive_path / "models"
        self.log_dir = self.drive_path / "logs"
        
        # Create necessary directories
        for dir_path in [self.checkpoint_dir, self.data_dir, self.model_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Set up logging to file
        log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    def save_model(self, model: torch.nn.Module, model_name: str, metadata: Dict = None):
        """Save model and metadata to Google Drive."""
        # Save model state
        model_path = self.model_dir / f"{model_name}.pth"
        torch.save(model.state_dict(), model_path)
        
        # Save metadata
        if metadata:
            metadata_path = self.model_dir / f"{model_name}_metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
        
        logger.info(f"Model saved to {model_path}")
        if metadata:
            logger.info(f"Metadata saved to {metadata_path}")
    
    def load_model(self, model_name: str, model_class: torch.nn.Module) -> Tuple[torch.nn.Module, Dict]:
        """Load model and metadata from Google Drive."""
        model_path = self.model_dir / f"{model_name}.pth"
        metadata_path = self.model_dir / f"{model_name}_metadata.pkl"
        
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return None, None
        
        # Load model
        model = model_class()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # Load metadata
        metadata = None
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
        
        logger.info(f"Model loaded from {model_path}")
        return model, metadata

def main():
    # Mount Google Drive
    drive.mount('/content/drive')
    
    # Initialize checkpoint manager
    checkpoint_manager = ColabCheckpoint()
    
    # Initialize loader
    loader = AmazonBooksLoader(checkpoint_manager)
    
    # Load data
    df = loader.load_data(max_records=10000)
    if df is None:
        logger.error("Failed to load data")
        return
    
    # Initialize trainer
    trainer = OptimizedAmazonTrainer(checkpoint_manager)
    
    # Train model
    model = trainer.train(df)
    
    # Save final model with metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'model_architecture': 'TimeSeriesDBN',
        'layer_sizes': [3, 64, 32, 16],
        'sequence_length': 10,
        'batch_size': 64,
        'learning_rate': 0.001,
        'device': str(trainer.device)
    }
    
    checkpoint_manager.save_model(model, "amazon_anomaly_detector", metadata)
    logger.info("Training completed and model saved to Google Drive")

if __name__ == "__main__":
    main() 