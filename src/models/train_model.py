import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from tqdm import tqdm
from datetime import datetime
from ..utils.review_visualizer import ReviewVisualizer
from .review_analyzer import ReviewAnalyzer

class ModelTrainer:
    def __init__(self, model_dir: str = "models/checkpoints"):
        """
        Initialize the model trainer.
        
        Args:
            model_dir (str): Directory to save model checkpoints
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def train(self, 
              train_data: pd.DataFrame,
              val_data: pd.DataFrame,
              batch_size: int = 32,
              epochs: int = 10,
              learning_rate: float = 0.001,
              checkpoint_freq: int = 1):
        """
        Train the review analyzer model.
        
        Args:
            train_data (pd.DataFrame): Training data
            val_data (pd.DataFrame): Validation data
            batch_size (int): Batch size for training
            epochs (int): Number of epochs to train
            learning_rate (float): Learning rate for optimizer
            checkpoint_freq (int): Frequency of saving checkpoints (in epochs)
        """
        # Initialize model and move to device
        model = ReviewAnalyzer()
        model.to(self.device)
        
        # Initialize optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # Create timestamp for this training run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = self.model_dir / timestamp
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            # Training phase
            for batch in tqdm(self._get_batches(train_data, batch_size), 
                            desc=f"Epoch {epoch + 1}/{epochs} [Train]"):
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(batch['text'])
                loss = criterion(outputs, batch['labels'])
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Update metrics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch['labels'].size(0)
                train_correct += (predicted == batch['labels']).sum().item()
            
            # Calculate training metrics
            avg_train_loss = train_loss / len(train_data)
            train_acc = 100 * train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in tqdm(self._get_batches(val_data, batch_size), 
                                desc=f"Epoch {epoch + 1}/{epochs} [Val]"):
                    outputs = model(batch['text'])
                    loss = criterion(outputs, batch['labels'])
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch['labels'].size(0)
                    val_correct += (predicted == batch['labels']).sum().item()
            
            # Calculate validation metrics
            avg_val_loss = val_loss / len(val_data)
            val_acc = 100 * val_correct / val_total
            
            # Update history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )
            
            # Save checkpoint
            if (epoch + 1) % checkpoint_freq == 0:
                self._save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    history=history,
                    checkpoint_dir=checkpoint_dir
                )
        
        # Save final model and training history
        self._save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epochs - 1,
            history=history,
            checkpoint_dir=checkpoint_dir,
            is_final=True
        )
        
        return history
    
    def _get_batches(self, data: pd.DataFrame, batch_size: int):
        """Generate batches of data for training."""
        for i in range(0, len(data), batch_size):
            batch_data = data.iloc[i:i + batch_size]
            yield {
                'text': batch_data['review_text'].values,
                'labels': torch.tensor(batch_data['is_anomaly'].values, 
                                     dtype=torch.long).to(self.device)
            }
    
    def _save_checkpoint(self, 
                        model: nn.Module,
                        optimizer: optim.Optimizer,
                        epoch: int,
                        history: dict,
                        checkpoint_dir: Path,
                        is_final: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history
        }
        
        # Save model checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save training history
        history_path = checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f)
        
        # Save final model
        if is_final:
            final_model_path = checkpoint_dir / "final_model.pt"
            torch.save(model.state_dict(), final_model_path)
            
            # Save model configuration
            config = {
                'model_type': model.__class__.__name__,
                'device': str(self.device),
                'timestamp': datetime.now().isoformat()
            }
            config_path = checkpoint_dir / "model_config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f)
        
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")

def main():
    """Main function to train the model."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # First ensure data is downloaded
    from ..data.download_data import DataDownloader
    downloader = DataDownloader()
    downloaded_files = downloader.download_all()
    
    # Load and prepare data
    from ..data.amazon_loader import AmazonBooksLoader
    loader = AmazonBooksLoader(downloaded_files['books'])
    
    # Load and split data
    books_df = loader.load_data(max_items=1000)
    reviews_df = loader.get_reviews(downloaded_files['reviews'], max_reviews=10000)
    merged_df = loader.merge_books_and_reviews(books_df, reviews_df)
    analysis_df = loader.prepare_for_analysis(merged_df)
    
    # Split data into train and validation sets
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(analysis_df, test_size=0.2, random_state=42)
    
    # Train model
    history = trainer.train(
        train_data=train_df,
        val_data=val_df,
        batch_size=32,
        epochs=10,
        learning_rate=0.001,
        checkpoint_freq=1
    )

if __name__ == "__main__":
    main() 