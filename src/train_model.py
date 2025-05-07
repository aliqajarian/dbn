import torch
from pathlib import Path
import json
import logging
from models.dbn import TimeSeriesDBN
from data.preprocessor import TextPreprocessor
from data.feature_engineering import FeatureEngineer
from utils.gan_visualization import GANVisualizer
from utils.gan_metrics import GANMetrics
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config/model_config.json"):
    with open(config_path, 'r') as f:
        return json.load(f)

def train_model():
    # Load configuration
    config = load_config()
    
    # Initialize components
    preprocessor = TextPreprocessor(
        max_features=config['data']['max_features'],
        min_df=config['data']['min_df']
    )
    feature_engineer = FeatureEngineer()
    visualizer = GANVisualizer(feature_names=preprocessor.vectorizer.get_feature_names_out())
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    df = pd.read_csv("data/raw/amazon_reviews.csv")
    features = preprocessor.preprocess(df['text'].values)
    behavioral_features = feature_engineer.extract_behavioral_features(df)
    combined_features = feature_engineer.combine_features(features, behavioral_features)
    
    # Create sequences
    sequences = feature_engineer.create_sequences(combined_features)
    
    # Initialize model
    model = TimeSeriesDBN(
        layer_sizes=config['model']['layer_sizes'],
        sequence_length=config['data']['sequence_length']
    )
    
    # Train model
    logger.info("Training model...")
    metrics = model.train(
        sequences,
        epochs=config['model']['epochs'],
        batch_size=config['model']['batch_size'],
        use_gan=True
    )
    
    # Save model
    torch.save(model.state_dict(), "models/anomaly_detector.pth")
    logger.info("Model saved to models/anomaly_detector.pth")
    
    return model, metrics

if __name__ == "__main__":
    train_model() 