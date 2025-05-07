import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from models.dbn import TimeSeriesDBN
from models.gan import Generator, Discriminator, train_gan
from data.preprocessor import TextPreprocessor
from data.feature_engineering import FeatureEngineer
from utils.visualization import plot_reconstruction_error, plot_roc_curve
from utils.metrics import compute_metrics
import pandas as pd
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyDetector:
    def __init__(self, 
                 layer_sizes: List[int],
                 sequence_length: int = 10,
                 learning_rate: float = 0.001,
                 noise_dim: int = 100):
        self.model = TimeSeriesDBN(layer_sizes, sequence_length)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.text_preprocessor = TextPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.noise_dim = noise_dim
        
        # Initialize GAN components
        self.generator = Generator(noise_dim, layer_sizes[0])
        self.discriminator = Discriminator(layer_sizes[0])
        
    def prepare_data(self, 
                    reviews: pd.DataFrame,
                    sequence_length: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data for training."""
        # Preprocess text
        text_features = self.text_preprocessor.preprocess(reviews['text'].values)
        
        # Extract behavioral features
        behavioral_features = self.feature_engineer.extract_behavioral_features(reviews)
        
        # Combine features
        combined_features = self.feature_engineer.combine_features(
            text_features, behavioral_features)
        
        # Create sequences
        sequences = []
        for i in range(len(combined_features) - sequence_length + 1):
            sequences.append(combined_features[i:i + sequence_length])
        
        return torch.FloatTensor(sequences)
    
    def generate_synthetic_data(self, n_samples: int) -> torch.Tensor:
        """Generate synthetic data using GAN."""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.noise_dim)
            synthetic_data = self.generator(z)
        return synthetic_data
    
    def train(self, 
              train_loader: DataLoader,
              epochs: int,
              anomaly_threshold: float = 0.1,
              use_gan: bool = True) -> Dict[str, float]:
        """Train the model with optional GAN enhancement."""
        self.model.train()
        total_loss = 0
        
        # Train GAN if enabled
        if use_gan:
            logger.info("Training GAN for synthetic data generation...")
            train_gan(self.generator, self.discriminator, train_loader, self.noise_dim)
            
            # Generate synthetic data
            synthetic_data = self.generate_synthetic_data(len(train_loader.dataset))
            synthetic_loader = DataLoader(TensorDataset(synthetic_data), 
                                        batch_size=train_loader.batch_size, 
                                        shuffle=True)
            
            # Combine real and synthetic data
            combined_loader = self._combine_loaders(train_loader, synthetic_loader)
        else:
            combined_loader = train_loader
        
        # Train DBN
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in combined_loader:
                data = batch[0]
                
                # Forward pass
                self.optimizer.zero_grad()
                reconstruction = self.model.reconstruct(data)
                
                # Calculate reconstruction error
                loss = torch.mean((data - reconstruction) ** 2)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(combined_loader)
            logger.info(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
            total_loss += avg_loss
        
        return {'avg_loss': total_loss / epochs}
    
    def _combine_loaders(self, 
                        real_loader: DataLoader, 
                        synthetic_loader: DataLoader) -> DataLoader:
        """Combine real and synthetic data loaders."""
        real_data = next(iter(real_loader))[0]
        synthetic_data = next(iter(synthetic_loader))[0]
        combined_data = torch.cat([real_data, synthetic_data], dim=0)
        return DataLoader(TensorDataset(combined_data), 
                         batch_size=real_loader.batch_size, 
                         shuffle=True)
    
    def evaluate(self, 
                test_loader: DataLoader,
                threshold: float) -> Dict[str, float]:
        """Evaluate model performance with visualization."""
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_errors = []
        
        with torch.no_grad():
            for batch in test_loader:
                data, labels = batch
                anomalies, errors = self.model.detect_anomalies(data, threshold)
                all_predictions.extend(anomalies.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_errors.extend(errors.cpu().numpy())
        
        # Compute metrics
        metrics = compute_metrics(np.array(all_labels), 
                                np.array(all_predictions), 
                                np.array(all_errors))
        
        # Visualize results
        plot_reconstruction_error(np.array(all_errors), threshold)
        plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['roc_auc'])
        
        return metrics

def main():
    # Example usage
    # Load your data
    # reviews_df = pd.read_csv('path_to_your_reviews.csv')
    
    # Initialize detector
    layer_sizes = [1000, 500, 200, 100]  # Adjust based on your feature size
    detector = AnomalyDetector(layer_sizes)
    
    # Prepare data
    # sequences = detector.prepare_data(reviews_df)
    # train_loader = DataLoader(TensorDataset(sequences), batch_size=32, shuffle=True)
    
    # Train model with GAN
    # metrics = detector.train(train_loader, epochs=50, use_gan=True)
    
    # Evaluate
    # eval_metrics = detector.evaluate(test_loader, threshold=0.1)
    
    # Save model
    # torch.save(detector.model.state_dict(), 'anomaly_detector.pth')

if __name__ == "__main__":
    main() 