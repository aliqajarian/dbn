import torch
from pathlib import Path
import logging
from data.amazon_loader import AmazonBooksLoader
from data.preprocessor import TextPreprocessor
from data.feature_engineering import FeatureEngineer
from models.dbn import TimeSeriesDBN
from utils.visualization import plot_reconstruction_error
from utils.metrics import compute_metrics
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_amazon_model():
    # Initialize data loader
    loader = AmazonBooksLoader()
    
    # Load and preprocess data
    df = loader.load_data()
    df = loader.preprocess_data(df)
    
    # Initialize preprocessor and feature engineer
    preprocessor = TextPreprocessor(max_features=1000)
    feature_engineer = FeatureEngineer()
    
    # Extract features
    logger.info("Extracting features...")
    text_features = preprocessor.preprocess(df['text'].values)
    
    # Create behavioral features
    behavioral_features = pd.DataFrame({
        'price': df['price'],
        'rating': df['rating'],
        'review_count': df['review_count'],
        'category_depth': df['category_depth'],
        'subcategory_count': df['subcategory_count']
    })
    
    # Combine features
    features = feature_engineer.combine_features(text_features, behavioral_features)
    
    # Create sequences for time series analysis
    sequences = feature_engineer.create_sequences(features)
    
    # Split data
    train_data, test_data = train_test_split(sequences, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    train_tensor = torch.FloatTensor(train_data)
    test_tensor = torch.FloatTensor(test_data)
    
    # Create data loaders
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_tensor), batch_size=32)
    
    # Initialize model
    model = TimeSeriesDBN(
        layer_sizes=[1000, 500, 200, 100],
        sequence_length=10
    )
    
    # Train model
    logger.info("Training model...")
    metrics = model.train(
        train_loader,
        epochs=50,
        learning_rate=0.001
    )
    
    # Evaluate model
    logger.info("Evaluating model...")
    with torch.no_grad():
        test_errors = model.get_reconstruction_error(test_tensor)
    
    # Plot reconstruction errors
    plot_reconstruction_error(test_errors.numpy(), threshold=0.1)
    
    # Save model
    torch.save(model.state_dict(), "models/amazon_anomaly_detector.pth")
    logger.info("Model saved to models/amazon_anomaly_detector.pth")
    
    return model, metrics

def detect_anomalies(model, data, threshold=0.1):
    """Detect anomalies in the data."""
    with torch.no_grad():
        errors = model.get_reconstruction_error(data)
        anomalies = errors > threshold
    return anomalies, errors

if __name__ == "__main__":
    model, metrics = train_amazon_model() 