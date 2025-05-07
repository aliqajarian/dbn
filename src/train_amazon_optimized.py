import torch
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import gc
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            layer_sizes=[1000, 500, 200, 100],
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
    
    def create_features(self, df):
        """Create features efficiently"""
        # Implement your feature creation logic here
        pass

def main():
    # Initialize trainer
    trainer = OptimizedAmazonTrainer()
    
    # Load data
    loader = AmazonBooksLoader()
    df = loader.load_data()
    
    # Train model
    model = trainer.train(df)
    
    # Save model
    torch.save(model.state_dict(), "models/amazon_anomaly_detector.pth")
    logger.info("Training completed and model saved")

if __name__ == "__main__":
    main() 