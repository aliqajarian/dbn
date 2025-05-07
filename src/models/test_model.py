import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from google.colab import drive
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

class ModelTester:
    def __init__(self, model_dir: str = "models/checkpoints", use_gdrive: bool = False):
        """
        Initialize the model tester.
        
        Args:
            model_dir (str): Directory containing model checkpoints
            use_gdrive (bool): Whether to use Google Drive for storage
        """
        self.model_dir = Path(model_dir)
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_gdrive = use_gdrive
        
        if use_gdrive:
            self._setup_gdrive()
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased').to(self.device)
        
        # Initialize comparison models
        self.comparison_models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'tfidf': TfidfVectorizer(max_features=1000)
        }
    
    def _setup_gdrive(self):
        """Set up Google Drive integration."""
        try:
            drive.mount('/content/drive')
            self.gdrive_dir = Path('/content/drive/MyDrive/amazon_reviews_model')
            self.gdrive_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Google Drive mounted at {self.gdrive_dir}")
        except Exception as e:
            self.logger.error(f"Error mounting Google Drive: {str(e)}")
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
            self.logger.error(f"Error saving to Google Drive: {str(e)}")
    
    def load_test_data(self, file_path: str, max_items: int = 1000) -> pd.DataFrame:
        """
        Load and process test data from the CSV file.
        
        Args:
            file_path (str): Path to the test CSV file
            max_items (int): Maximum number of items to load
            
        Returns:
            pd.DataFrame: Processed test data
        """
        self.logger.info(f"Loading test data from {file_path}")
        
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
                'Id': 'asin',
                'ProductId': 'parent_asin',
                'UserId': 'user_id',
                'Score': 'rating',
                'Summary': 'title',
                'Text': 'text',
                'HelpfulnessNumerator': 'helpful_votes',
                'Time': 'sort_timestamp'
            }
            
            # Rename columns if they exist
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df = df.rename(columns={old_col: new_col})
            
            # Process the data with available columns
            processed_data = {}
            
            # Handle required columns
            for col in ['asin', 'parent_asin', 'user_id', 'rating', 'title', 'text', 'helpful_votes', 'sort_timestamp']:
                if col in df.columns:
                    if col == 'rating':
                        processed_data[col] = df[col].astype(float)
                    elif col == 'helpful_votes':
                        processed_data[col] = df[col].fillna(0).astype(int)
                    elif col == 'sort_timestamp':
                        processed_data[col] = df[col].fillna(0).astype(int)
                    else:
                        processed_data[col] = df[col].fillna('')
                else:
                    self.logger.warning(f"Column {col} not found in CSV file")
                    # Add default values for missing columns
                    if col == 'rating':
                        processed_data[col] = pd.Series(0.0, index=df.index)
                    elif col == 'helpful_votes':
                        processed_data[col] = pd.Series(0, index=df.index)
                    elif col == 'sort_timestamp':
                        processed_data[col] = pd.Series(0, index=df.index)
                    else:
                        processed_data[col] = pd.Series('', index=df.index)
            
            # Add verified_purchase column (not in the dataset, so default to False)
            processed_data['verified_purchase'] = pd.Series(False, index=df.index)
            
            processed_df = pd.DataFrame(processed_data)
            
            self.logger.info(f"Loaded {len(processed_df)} test reviews")
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Error loading test data: {str(e)}")
            raise
    
    def prepare_test_data(self, df: pd.DataFrame) -> tuple:
        """
        Prepare test data for evaluation.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            tuple: (X, y) where X is the input features and y is the target
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
    
    def load_model(self, checkpoint_path: str):
        """
        Load a trained model from checkpoint.
        
        Args:
            checkpoint_path (str): Path to the model checkpoint
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.logger.info(f"Loaded model from {checkpoint_path}")
            
            if self.use_gdrive:
                self.save_to_gdrive('model_checkpoints/final_model.pt', checkpoint_path)
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def evaluate(self, test_data: tuple) -> dict:
        """
        Evaluate the model on test data.
        
        Args:
            test_data (tuple): (X_test, y_test, texts)
            
        Returns:
            dict: Evaluation metrics
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(test_data[0][0]), 32), desc="Evaluating"):
                # Get batch
                batch_input_ids = test_data[0][0][i:i + 32]
                batch_attention_mask = test_data[0][1][i:i + 32]
                batch_labels = test_data[1][i:i + 32]
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask
                )
                logits = outputs.last_hidden_state[:, 0, :]
                probs = torch.softmax(logits, dim=1)
                _, predicted = torch.max(logits.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        report = classification_report(all_labels, all_predictions, output_dict=True)
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        
        # Create visualizations
        self._create_visualizations(all_labels, all_predictions, all_probs, conf_matrix)
        
        return {
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': all_predictions,
            'probabilities': all_probs
        }
    
    def _create_visualizations(self, labels, predictions, probabilities, conf_matrix):
        """Create various visualizations of the results."""
        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(self.model_dir / "confusion_matrix.png")
        plt.close()
        
        # 2. ROC Curve
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(labels, [p[1] for p in probabilities])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(self.model_dir / "roc_curve.png")
        plt.close()
        
        # 3. Interactive Plotly Dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Confusion Matrix', 'ROC Curve', 'Prediction Distribution', 'Feature Importance')
        )
        
        # Confusion Matrix
        fig.add_trace(
            go.Heatmap(z=conf_matrix, colorscale='Blues', showscale=False),
            row=1, col=1
        )
        
        # ROC Curve
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, name=f'ROC (AUC = {roc_auc:.2f})'),
            row=1, col=2
        )
        
        # Prediction Distribution
        fig.add_trace(
            go.Histogram(x=predictions, name='Predictions'),
            row=2, col=1
        )
        
        # Save interactive plot
        fig.write_html(str(self.model_dir / "interactive_results.html"))
        
        if self.use_gdrive:
            self.save_to_gdrive('visualizations/confusion_matrix.png', self.model_dir / "confusion_matrix.png")
            self.save_to_gdrive('visualizations/roc_curve.png', self.model_dir / "roc_curve.png")
            self.save_to_gdrive('visualizations/interactive_results.html', self.model_dir / "interactive_results.html")
    
    def compare_with_baselines(self, test_data: tuple) -> dict:
        """
        Compare BERT model with baseline models.
        
        Args:
            test_data (tuple): (X_test, y_test, texts)
            
        Returns:
            dict: Comparison results
        """
        X_test, y_test, texts = test_data
        results = {}
        
        # 1. BERT Model (already evaluated)
        bert_results = self.evaluate(test_data)
        results['bert'] = bert_results['classification_report']
        
        # 2. Random Forest
        rf = self.comparison_models['random_forest']
        tfidf = self.comparison_models['tfidf']
        
        # Transform text data
        X_tfidf = tfidf.fit_transform(texts)
        
        # Train and evaluate RF
        rf.fit(X_tfidf, y_test.cpu().numpy())
        rf_pred = rf.predict(X_tfidf)
        results['random_forest'] = classification_report(y_test.cpu().numpy(), rf_pred, output_dict=True)
        
        # Create comparison visualization
        self._create_comparison_visualization(results)
        
        return results
    
    def _create_comparison_visualization(self, results: dict):
        """Create visualization comparing different models."""
        # Extract metrics for comparison
        models = list(results.keys())
        metrics = ['precision', 'recall', 'f1-score']
        
        # Create comparison plot
        fig = go.Figure()
        
        for metric in metrics:
            values = [results[model]['weighted avg'][metric] for model in models]
            fig.add_trace(go.Bar(
                name=metric,
                x=models,
                y=values,
                text=[f'{v:.3f}' for v in values],
                textposition='auto',
            ))
        
        fig.update_layout(
            title='Model Comparison',
            xaxis_title='Models',
            yaxis_title='Score',
            barmode='group'
        )
        
        # Save plot
        fig.write_html(str(self.model_dir / "model_comparison.html"))
        
        if self.use_gdrive:
            self.save_to_gdrive('visualizations/model_comparison.html', self.model_dir / "model_comparison.html")
    
    def save_results(self, results: dict, output_path: str):
        """
        Save evaluation results to file.
        
        Args:
            results (dict): Evaluation results
            output_path (str): Path to save results
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=4)
            self.logger.info(f"Saved results to {output_path}")
            
            if self.use_gdrive:
                self.save_to_gdrive('results/test_results.json', results)
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise

def main():
    """Main function to test the model."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize tester with Google Drive integration
    tester = ModelTester(use_gdrive=True)
    
    try:
        # Load test data
        test_df = tester.load_test_data("data/raw/Books_5.csv", max_items=1000)
        
        # Prepare test data
        X_test, y_test, texts = tester.prepare_test_data(test_df)
        
        # Load latest model checkpoint
        checkpoint_dir = Path("models/checkpoints")
        if not checkpoint_dir.exists():
            raise FileNotFoundError("No model checkpoints found")
        
        # Find latest checkpoint
        checkpoints = list(checkpoint_dir.glob("**/final_model.pt"))
        if not checkpoints:
            raise FileNotFoundError("No final model found")
        
        latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
        tester.load_model(str(latest_checkpoint))
        
        # Evaluate model and compare with baselines
        results = tester.compare_with_baselines((X_test, y_test, texts))
        
        # Save results
        results_path = checkpoint_dir / "test_results.json"
        tester.save_results(results, str(results_path))
        
        # Print results
        print("\nTest Results:")
        print(json.dumps(results['bert']['classification_report'], indent=4))
        print("\nModel Comparison:")
        print(json.dumps({k: v['weighted avg'] for k, v in results.items()}, indent=4))
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 