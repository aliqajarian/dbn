import torch
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import gc
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import requests
import gzip
import json
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AmazonBooksLoader:
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.url = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Books.jsonl.gz"
    
    def download_data(self):
        """Download the Amazon Books dataset."""
        logger.info("Downloading Amazon Books dataset...")
        try:
            response = requests.get(self.url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            file_path = self.data_dir / "meta_Books.jsonl.gz"
            with open(file_path, 'wb') as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit='iB',
                unit_scale=True
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
            
            logger.info(f"Dataset downloaded to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            return None
    
    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """Load and process the Amazon Books dataset."""
        if file_path is None:
            file_path = self.data_dir / "meta_Books.jsonl.gz"
        
        if not file_path.exists():
            file_path = self.download_data()
        
        logger.info("Loading and processing data...")
        data = []
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line in tqdm(f, desc="Processing records"):
                    try:
                        record = json.loads(line)
                        # Extract relevant fields
                        processed_record = {
                            'asin': record.get('asin', ''),
                            'title': record.get('title', ''),
                            'description': record.get('description', ''),
                            'price': float(record.get('price', 0.0)),
                            'rating': float(record.get('rating', 0.0)),
                            'review_count': int(record.get('review_count', 0))
                        }
                        data.append(processed_record)
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning(f"Error processing record: {e}")
                        continue
            
            df = pd.DataFrame(data)
            logger.info(f"Loaded {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None

class TimeSeriesDBN(torch.nn.Module):
    def __init__(self, layer_sizes: List[int], sequence_length: int = 10):
        super(TimeSeriesDBN, self).__init__()
        self.layer_sizes = layer_sizes
        self.sequence_length = sequence_length
        
        # Create layers
        self.layers = torch.nn.ModuleList()
        prev_size = layer_sizes[0]
        for size in layer_sizes[1:]:
            self.layers.append(torch.nn.Linear(prev_size, size))
            prev_size = size
    
    def forward(self, x):
        for layer in self.layers:
            x = torch.sigmoid(layer(x))
        return x
    
    def reconstruct(self, x):
        return self.forward(x)
    
    def get_reconstruction_error(self, x):
        with torch.no_grad():
            reconstruction = self.reconstruct(x)
            return torch.mean((x - reconstruction) ** 2, dim=1)

class OptimizedAmazonTrainer:
    def __init__(self, batch_size=64, sequence_length=10):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
    def prepare_data(self, df):
        """Optimized data preparation"""
        logger.info("Preparing data...")
        
        # Reduce memory usage
        df['text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
        df = df[['text', 'price', 'rating', 'review_count']]
        
        # Convert to float32 to reduce memory
        df['price'] = df['price'].astype('float32')
        df['rating'] = df['rating'].astype('float32')
        df['review_count'] = df['review_count'].astype('float32')
        
        return df
    
    def create_features(self, df):
        """Create features efficiently"""
        # Normalize numerical features
        features = df[['price', 'rating', 'review_count']].values
        features = (features - features.mean(axis=0)) / features.std(axis=0)
        return features
    
    def train(self, df, epochs=50):
        """Optimized training process"""
        logger.info("Starting training...")
        
        # Prepare data
        df = self.prepare_data(df)
        
        # Create features
        features = self.create_features(df)
        
        # Split data
        train_data, test_data = train_test_split(features, test_size=0.2, random_state=42)
        
        # Create data loaders
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(train_data)),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # Initialize model
        model = TimeSeriesDBN(
            layer_sizes=[3, 64, 32, 16],  # Adjusted for our feature size
            sequence_length=self.sequence_length
        ).to(self.device)
        
        # Training loop
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                data = batch[0].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                reconstruction = model.reconstruct(data)
                loss = torch.mean((data - reconstruction) ** 2)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            # Clear memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        return model

def main():
    # Initialize loader
    loader = AmazonBooksLoader()
    
    # Load data
    df = loader.load_data()
    if df is None:
        logger.error("Failed to load data")
        return
    
    # Initialize trainer
    trainer = OptimizedAmazonTrainer()
    
    # Train model
    model = trainer.train(df)
    
    # Save model
    torch.save(model.state_dict(), "models/amazon_anomaly_detector.pth")
    logger.info("Training completed and model saved")

if __name__ == "__main__":
    main() 