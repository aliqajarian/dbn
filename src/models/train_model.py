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
import gzip
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel

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
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased').to(self.device)
        
    def load_books_data(self, file_path: str, max_items: int = 1000) -> pd.DataFrame:
        """
        Load and process books data from the meta_Books.jsonl.gz file.
        
        Args:
            file_path (str): Path to the meta_Books.jsonl.gz file
            max_items (int): Maximum number of items to load
            
        Returns:
            pd.DataFrame: Processed books data
        """
        self.logger.info(f"Loading data from {file_path}")
        data = []
        
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Loading books")):
                if i >= max_items:
                    break
                    
                try:
                    item = json.loads(line.strip())
                    # Extract relevant fields for analysis
                    processed_item = {
                        'asin': item.get('parent_asin', ''),
                        'title': item.get('title', ''),
                        'description': ' '.join(item.get('description', [])),
                        'categories': item.get('categories', []),
                        'price': float(item.get('price', 0.0)),
                        'average_rating': float(item.get('average_rating', 0.0)),
                        'rating_count': int(item.get('rating_number', 0)),
                        'main_category': item.get('main_category', ''),
                        'features': item.get('features', []),
                        'author': item.get('author', {}).get('name', '') if item.get('author') else ''
                    }
                    data.append(processed_item)
                except Exception as e:
                    self.logger.warning(f"Error processing item at line {i}: {e}")
                    continue
        
        df = pd.DataFrame(data)
        self.logger.info(f"Loaded {len(df)} items")
        return df
    
    def prepare_data(self, df: pd.DataFrame) -> tuple:
        """
        Prepare data for training.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            tuple: (X, y) where X is the input features and y is the target
        """
        # Combine text features
        df['text'] = df.apply(lambda x: f"Title: {x['title']} Description: {x['description']} Features: {' '.join(x['features'])}", axis=1)
        
        # Create target variable (example: predict if book is highly rated)
        df['is_highly_rated'] = (df['average_rating'] >= 4.0).astype(int)
        
        # Tokenize text
        encodings = self.tokenizer(
            df['text'].tolist(),
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        labels = torch.tensor(df['is_highly_rated'].values, dtype=torch.long).to(self.device)
        
        return (input_ids, attention_mask), labels
    
    def train(self, 
              train_data: tuple,
              val_data: tuple,
              batch_size: int = 32,
              epochs: int = 10,
              learning_rate: float = 0.001,
              checkpoint_freq: int = 1):
        """
        Train the model.
        
        Args:
            train_data (tuple): (X_train, y_train)
            val_data (tuple): (X_val, y_val)
            batch_size (int): Batch size for training
            epochs (int): Number of epochs to train
            learning_rate (float): Learning rate for optimizer
            checkpoint_freq (int): Frequency of saving checkpoints (in epochs)
        """
        # Initialize optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
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
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            # Training phase
            for i in tqdm(range(0, len(train_data[0][0]), batch_size), desc=f"Epoch {epoch + 1}/{epochs} [Train]"):
                # Get batch
                batch_input_ids = train_data[0][0][i:i + batch_size]
                batch_attention_mask = train_data[0][1][i:i + batch_size]
                batch_labels = train_data[1][i:i + batch_size]
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask
                )
                logits = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
                loss = criterion(logits, batch_labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Update metrics
                train_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()
            
            # Calculate training metrics
            avg_train_loss = train_loss / len(train_data[0][0])
            train_acc = 100 * train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for i in tqdm(range(0, len(val_data[0][0]), batch_size), desc=f"Epoch {epoch + 1}/{epochs} [Val]"):
                    # Get batch
                    batch_input_ids = val_data[0][0][i:i + batch_size]
                    batch_attention_mask = val_data[0][1][i:i + batch_size]
                    batch_labels = val_data[1][i:i + batch_size]
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask
                    )
                    logits = outputs.last_hidden_state[:, 0, :]
                    loss = criterion(logits, batch_labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(logits.data, 1)
                    val_total += batch_labels.size(0)
                    val_correct += (predicted == batch_labels).sum().item()
            
            # Calculate validation metrics
            avg_val_loss = val_loss / len(val_data[0][0])
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
                    model=self.model,
                    optimizer=optimizer,
                    epoch=epoch,
                    history=history,
                    checkpoint_dir=checkpoint_dir
                )
        
        # Save final model and training history
        self._save_checkpoint(
            model=self.model,
            optimizer=optimizer,
            epoch=epochs - 1,
            history=history,
            checkpoint_dir=checkpoint_dir,
            is_final=True
        )
        
        return history
    
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
    
    # Load and prepare data
    books_df = trainer.load_books_data("data/raw/meta_Books.jsonl.gz", max_items=1000)
    
    # Prepare data for training
    X, y = trainer.prepare_data(books_df)
    
    # Split data into train and validation sets
    train_size = int(0.8 * len(X[0]))
    train_data = (
        (X[0][:train_size], X[1][:train_size]),
        y[:train_size]
    )
    val_data = (
        (X[0][train_size:], X[1][train_size:]),
        y[train_size:]
    )
    
    # Train model
    history = trainer.train(
        train_data=train_data,
        val_data=val_data,
        batch_size=32,
        epochs=10,
        learning_rate=0.001,
        checkpoint_freq=1
    )

if __name__ == "__main__":
    main() 