import torch
import pandas as pd
import numpy as np
import gzip
import json
from pathlib import Path
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MinimalAmazonLoader:
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.url = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Books.jsonl.gz"
    
    def download_data(self):
        """Download a small sample of data for testing"""
        logger.info("Downloading sample data...")
        try:
            import requests
            response = requests.get(self.url, stream=True)
            file_path = self.data_dir / "meta_Books.jsonl.gz"
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"Data downloaded to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            return None

    def load_sample_data(self, n_samples=1000):
        """Load a small sample of data for testing"""
        try:
            file_path = self.data_dir / "meta_Books.jsonl.gz"
            if not file_path.exists():
                file_path = self.download_data()
            
            data = []
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for i, line in enumerate(tqdm(f, desc="Loading data")):
                    if i >= n_samples:
                        break
                    try:
                        record = json.loads(line)
                        processed_record = {
                            'asin': record.get('asin', ''),
                            'title': record.get('title', ''),
                            'description': record.get('description', ''),
                            'price': float(record.get('price', 0.0)),
                            'rating': float(record.get('rating', 0.0)),
                            'review_count': int(record.get('review_count', 0))
                        }
                        data.append(processed_record)
                    except json.JSONDecodeError:
                        continue
            
            df = pd.DataFrame(data)
            logger.info(f"Loaded {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None

class MinimalDBN(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(MinimalDBN, self).__init__()
        self.layers = torch.nn.ModuleList()
        
        # Create layers
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(torch.nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
    
    def forward(self, x):
        for layer in self.layers:
            x = torch.sigmoid(layer(x))
        return x
    
    def reconstruct(self, x):
        return self.forward(x)

def test_minimal_implementation():
    # Initialize loader
    loader = MinimalAmazonLoader()
    
    # Load sample data
    df = loader.load_sample_data(n_samples=1000)
    if df is None:
        logger.error("Failed to load data")
        return
    
    # Prepare features
    features = df[['price', 'rating', 'review_count']].values
    features = (features - features.mean(axis=0)) / features.std(axis=0)
    
    # Convert to PyTorch tensor
    features_tensor = torch.FloatTensor(features)
    
    # Initialize model
    model = MinimalDBN(
        input_size=features.shape[1],
        hidden_sizes=[64, 32, 16]
    )
    
    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 10
    
    logger.info("Starting training...")
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        
        # Forward pass
        reconstruction = model.reconstruct(features_tensor)
        loss = torch.mean((features_tensor - reconstruction) ** 2)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        logger.info(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")
    
    # Test reconstruction
    model.eval()
    with torch.no_grad():
        reconstruction = model.reconstruct(features_tensor)
        reconstruction_error = torch.mean((features_tensor - reconstruction) ** 2, dim=1)
        
        # Find anomalies
        threshold = reconstruction_error.mean() + 2 * reconstruction_error.std()
        anomalies = reconstruction_error > threshold
        
        logger.info(f"Found {anomalies.sum().item()} anomalies")
        
        # Print some examples
        if anomalies.sum().item() > 0:
            anomalous_indices = torch.where(anomalies)[0]
            for idx in anomalous_indices[:5]:
                logger.info(f"\nAnomaly found:")
                logger.info(f"Title: {df.iloc[idx]['title']}")
                logger.info(f"Price: ${df.iloc[idx]['price']:.2f}")
                logger.info(f"Rating: {df.iloc[idx]['rating']:.1f}")
                logger.info(f"Review Count: {df.iloc[idx]['review_count']}")
                logger.info(f"Reconstruction Error: {reconstruction_error[idx]:.4f}")

if __name__ == "__main__":
    test_minimal_implementation() 