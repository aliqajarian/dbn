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
from transformers import AutoTokenizer, AutoModel
import os
from google.colab import drive
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self, model_dir: str = "models/checkpoints", use_gdrive: bool = False):
        """
        Initialize the model trainer.
        
        Args:
            model_dir (str): Directory to save model checkpoints
            use_gdrive (bool): Whether to use Google Drive for storage
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_gdrive = use_gdrive
        
        if use_gdrive:
            self._setup_gdrive()
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased').to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=2e-5)
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def _setup_gdrive(self):
        """Set up Google Drive integration."""
        try:
            # Check if we're in a Colab environment
            import google.colab
            drive.mount('/content/drive')
            self.gdrive_dir = Path('/content/drive/MyDrive/amazon_reviews_model')
            self.gdrive_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Google Drive mounted at {self.gdrive_dir}")
        except (ImportError, Exception) as e:
            self.logger.warning(f"Google Drive mounting not available: {str(e)}")
            self.use_gdrive = False
    
    def save_to_gdrive(self, file_path: str, content: dict = None):
        """
        Save file to Google Drive.
        
        Args:
            file_path (str): Path to save the file
            content (dict): Content to save (if saving JSON)
        """
        if not self.use_gdrive:
            return
        
        try:
            gdrive_path = self.gdrive_dir / file_path
            gdrive_path.parent.mkdir(parents=True, exist_ok=True)
            
            if content is not None:
                with open(gdrive_path, 'w') as f:
                    json.dump(content, f, indent=4)
            else:
                # Copy file to Google Drive
                import shutil
                shutil.copy2(file_path, gdrive_path)
            
            self.logger.info(f"Saved to Google Drive: {gdrive_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save to Google Drive: {str(e)}")
    
    def load_reviews_data(self, file_path: str, max_items: int = 1000) -> pd.DataFrame:
        """
        Load and process reviews data from the CSV file.
        
        Args:
            file_path (str): Path to the reviews CSV file
            max_items (int): Maximum number of items to load
            
        Returns:
            pd.DataFrame: Processed reviews data
        """
        self.logger.info(f"Loading reviews data from {file_path}")
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                # Try to find the file in the data directory
                data_dir = Path("data/raw")
                if data_dir.exists():
                    files = list(data_dir.glob("*.csv"))
                    if files:
                        file_path = str(files[0])
                        self.logger.info(f"Found CSV file: {file_path}")
                    else:
                        raise FileNotFoundError(f"No CSV files found in {data_dir}")
                else:
                    raise FileNotFoundError(f"Data directory {data_dir} does not exist")
            
            # Read CSV file
            df = pd.read_csv(file_path, nrows=max_items)
            
            # Print column names for debugging
            self.logger.info(f"CSV columns: {df.columns.tolist()}")
            
            # Map column names to expected format
            column_mapping = {
                'Title': 'title',
                'description': 'text',
                'authors': 'author',
                'ratingsCount': 'rating',
                'categories': 'category'
            }
            
            # Rename columns if they exist
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df = df.rename(columns={old_col: new_col})
            
            # Process the data with available columns
            processed_data = {}
            
            # Handle required columns
            for col in ['title', 'text', 'rating']:
                if col in df.columns:
                    if col == 'rating':
                        processed_data[col] = df[col].fillna(0).astype(float)
                    else:
                        processed_data[col] = df[col].fillna('')
                else:
                    self.logger.warning(f"Column {col} not found in CSV file")
                    # Add default values for missing columns
                    if col == 'rating':
                        processed_data[col] = pd.Series(0.0, index=df.index)
                    else:
                        processed_data[col] = pd.Series('', index=df.index)
            
            # Add required columns for compatibility
            processed_data['asin'] = pd.Series([f'book_{i}' for i in range(len(df))], index=df.index)
            processed_data['parent_asin'] = processed_data['asin']
            processed_data['user_id'] = pd.Series([f'user_{i}' for i in range(len(df))], index=df.index)
            processed_data['helpful_votes'] = pd.Series(0, index=df.index)
            processed_data['sort_timestamp'] = pd.Series(0, index=df.index)
            processed_data['verified_purchase'] = pd.Series(False, index=df.index)
            
            processed_df = pd.DataFrame(processed_data)
            
            self.logger.info(f"Loaded {len(processed_df)} reviews")
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Error loading reviews data: {str(e)}")
            raise
    
    def prepare_data(self, df: pd.DataFrame) -> tuple:
        """
        Prepare data for training.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            tuple: (X, y, texts) where X is the input features, y is the target, and texts is the raw text
        """
        # Combine text features
        df['text'] = df.apply(lambda x: f"Title: {x['title']} Review: {x['text']}", axis=1)
        
        # Create target variable (example: predict if review is helpful)
        df['is_helpful'] = (df['helpful_votes'] > 0).astype(int)
        
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
        labels = torch.tensor(df['is_helpful'].values, dtype=torch.long).to(self.device)
        
        return (input_ids, attention_mask), labels, df['text'].tolist()
    
    def train(self, train_data: tuple, val_data: tuple = None, epochs: int = 3, batch_size: int = 32):
        """
        Train the model.
        
        Args:
            train_data (tuple): (X_train, y_train, texts_train)
            val_data (tuple): (X_val, y_val, texts_val)
            epochs (int): Number of epochs to train
            batch_size (int): Batch size for training
        """
        X_train, y_train, texts = train_data
        self.model.train()
        
        # Create timestamp for this training run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = self.model_dir / timestamp
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(epochs):
            self.logger.info(f"Epoch {epoch + 1}/{epochs}")
            total_loss = 0
            correct = 0
            total = 0
            
            # Training loop
            for i in tqdm(range(0, len(X_train[0]), batch_size), desc="Training"):
                # Get batch
                batch_input_ids = X_train[0][i:i + batch_size]
                batch_attention_mask = X_train[1][i:i + batch_size]
                batch_labels = y_train[i:i + batch_size]
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask
                )
                logits = outputs.last_hidden_state[:, 0, :]
                loss = self.criterion(logits, batch_labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Calculate accuracy
                _, predicted = torch.max(logits.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
                total_loss += loss.item()
            
            # Calculate epoch metrics
            epoch_loss = total_loss / (len(X_train[0]) / batch_size)
            epoch_acc = correct / total
            
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc)
            
            self.logger.info(f"Epoch {epoch + 1} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
            
            # Validation
            if val_data is not None:
                val_loss, val_acc = self._validate(val_data, batch_size)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                self.logger.info(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': epoch_loss,
                'accuracy': epoch_acc
            }
            
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            torch.save(checkpoint, checkpoint_path)
            
            if self.use_gdrive:
                try:
                    gdrive_checkpoint_path = self.gdrive_dir / "checkpoints" / timestamp / f"checkpoint_epoch_{epoch + 1}.pt"
                    gdrive_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(checkpoint, gdrive_checkpoint_path)
                except Exception as e:
                    self.logger.warning(f"Failed to save checkpoint to Google Drive: {str(e)}")
        
        # Save final model
        final_model_path = checkpoint_dir / "final_model.pt"
        torch.save(self.model.state_dict(), final_model_path)
        
        if self.use_gdrive:
            try:
                gdrive_final_path = self.gdrive_dir / "checkpoints" / timestamp / "final_model.pt"
                gdrive_final_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), gdrive_final_path)
            except Exception as e:
                self.logger.warning(f"Failed to save final model to Google Drive: {str(e)}")
        
        # Save training history
        history_path = checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        
        if self.use_gdrive:
            try:
                gdrive_history_path = self.gdrive_dir / "checkpoints" / timestamp / "training_history.json"
                gdrive_history_path.parent.mkdir(parents=True, exist_ok=True)
                with open(gdrive_history_path, 'w') as f:
                    json.dump(history, f, indent=4)
            except Exception as e:
                self.logger.warning(f"Failed to save training history to Google Drive: {str(e)}")
    
    def _validate(self, val_data: tuple, batch_size: int) -> tuple:
        """
        Validate the model.
        
        Args:
            val_data (tuple): (X_val, y_val, texts_val)
            batch_size (int): Batch size for validation
            
        Returns:
            tuple: (val_loss, val_acc)
        """
        X_val, y_val, _ = val_data
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i in range(0, len(X_val[0]), batch_size):
                # Get batch
                batch_input_ids = X_val[0][i:i + batch_size]
                batch_attention_mask = X_val[1][i:i + batch_size]
                batch_labels = y_val[i:i + batch_size]
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask
                )
                logits = outputs.last_hidden_state[:, 0, :]
                loss = self.criterion(logits, batch_labels)
                
                # Calculate accuracy
                _, predicted = torch.max(logits.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
                total_loss += loss.item()
        
        val_loss = total_loss / (len(X_val[0]) / batch_size)
        val_acc = correct / total
        
        return val_loss, val_acc

def main():
    """Main function to train the model."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize trainer with Google Drive integration
    trainer = ModelTrainer(use_gdrive=True)
    
    try:
        # Load and prepare data
        train_df = trainer.load_reviews_data("data/raw/Books_5.csv", max_items=1000)
        train_data = trainer.prepare_data(train_df)
        
        # Train model
        trainer.train(train_data, epochs=3, batch_size=32)
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 