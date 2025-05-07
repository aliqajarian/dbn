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
    
    def safe_float_conversion(self, value, default=0.0):
        """Safely convert a value to float."""
        try:
            if value is None:
                return default
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def safe_int_conversion(self, value, default=0):
        """Safely convert a value to integer."""
        try:
            if value is None:
                return default
            return int(float(value))
        except (ValueError, TypeError):
            return default
    
    def process_record(self, record):
        """Process a single record with error handling."""
        try:
            # Extract and validate fields
            price = self.safe_float_conversion(record.get('price'))
            rating = self.safe_float_conversion(record.get('rating'))
            review_count = self.safe_int_conversion(record.get('review_count'))
            
            # Validate text fields
            title = str(record.get('title', '')).strip()
            description = str(record.get('description', '')).strip()
            
            # Only include records with valid data
            if price >= 0 and 0 <= rating <= 5 and review_count >= 0:
                return {
                    'asin': str(record.get('asin', '')),
                    'title': title,
                    'description': description,
                    'price': price,
                    'rating': rating,
                    'review_count': review_count
                }
            return None
        except Exception as e:
            logger.debug(f"Error processing record: {e}")
            return None
    
    def load_data(self, file_path: str = None, max_records: int = None) -> pd.DataFrame:
        """Load and process the Amazon Books dataset with improved error handling."""
        if file_path is None:
            file_path = self.data_dir / "meta_Books.jsonl.gz"
        
        if not file_path.exists():
            file_path = self.download_data()
        
        logger.info("Loading and processing data...")
        data = []
        processed_count = 0
        error_count = 0
        
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line in tqdm(f, desc="Processing records"):
                    try:
                        record = json.loads(line)
                        processed_record = self.process_record(record)
                        
                        if processed_record is not None:
                            data.append(processed_record)
                            processed_count += 1
                        else:
                            error_count += 1
                        
                        # Check if we've reached the maximum number of records
                        if max_records and processed_count >= max_records:
                            break
                            
                    except json.JSONDecodeError:
                        error_count += 1
                        continue
                    except Exception as e:
                        logger.warning(f"Unexpected error processing record: {e}")
                        error_count += 1
                        continue
            
            if not data:
                logger.error("No valid records found in the dataset")
                return None
            
            df = pd.DataFrame(data)
            
            # Log statistics
            logger.info(f"Successfully processed {processed_count} records")
            logger.info(f"Encountered {error_count} errors")
            logger.info(f"Final dataset shape: {df.shape}")
            
            # Log data statistics
            logger.info("\nData Statistics:")
            logger.info(f"Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
            logger.info(f"Rating range: {df['rating'].min():.1f} - {df['rating'].max():.1f}")
            logger.info(f"Review count range: {df['review_count'].min()} - {df['review_count'].max()}")
            
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
        
        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Reduce memory usage
        df.loc[:, 'text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
        df = df[['text', 'price', 'rating', 'review_count']].copy()
        
        # Convert to float32 to reduce memory using .loc
        df.loc[:, 'price'] = df['price'].astype('float32')
        df.loc[:, 'rating'] = df['rating'].astype('float32')
        df.loc[:, 'review_count'] = df['review_count'].astype('float32')
        
        return df
    
    def create_features(self, df):
        """Create features efficiently"""
        # Normalize numerical features
        features = df[['price', 'rating', 'review_count']].values
        
        # Handle potential NaN values and zero standard deviation
        mean = np.nanmean(features, axis=0)
        std = np.nanstd(features, axis=0)
        std[std == 0] = 1  # Prevent division by zero
        
        features = (features - mean) / std
        
        # Replace any remaining NaN values with 0
        features = np.nan_to_num(features, 0)
        
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
        
        # Initialize model with correct input size
        input_size = features.shape[1]  # This will be 3 for our features
        model = TimeSeriesDBN(
            layer_sizes=[input_size, 64, 32, input_size],  # Ensure output size matches input
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
    
    # Load data with a limit for testing
    df = loader.load_data(max_records=10000)  # Limit to 10,000 records for testing
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