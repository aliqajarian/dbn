import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from tqdm import tqdm
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from google.colab import drive
import warnings
warnings.filterwarnings('ignore')

class ResultsVisualizer:
    def __init__(self, model_dir: str = "models/checkpoints", use_gdrive: bool = False):
        """
        Initialize the results visualizer.
        
        Args:
            model_dir (str): Directory containing model checkpoints and results
            use_gdrive (bool): Whether to use Google Drive for storage
        """
        self.model_dir = Path(model_dir)
        self.logger = logging.getLogger(__name__)
        self.use_gdrive = use_gdrive
        
        if use_gdrive:
            self._setup_gdrive()
        
        # Initialize comparison models
        self.comparison_models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'naive_bayes': MultinomialNB(),
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
    
    def visualize_training_progress(self, history: dict, output_dir: Path):
        """
        Create visualizations of training progress.
        
        Args:
            history (dict): Training history containing metrics
            output_dir (Path): Directory to save visualizations
        """
        # 1. Interactive Dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Loss', 'Accuracy', 'Learning Rate', 'Metrics Comparison')
        )
        
        # Loss plot
        fig.add_trace(
            go.Scatter(y=history['train_loss'], name='Train Loss'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(y=history['val_loss'], name='Val Loss'),
            row=1, col=1
        )
        
        # Accuracy plot
        fig.add_trace(
            go.Scatter(y=history['train_acc'], name='Train Acc'),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(y=history['val_acc'], name='Val Acc'),
            row=1, col=2
        )
        
        # Learning rate plot
        if 'learning_rates' in history:
            fig.add_trace(
                go.Scatter(y=history['learning_rates'], name='Learning Rate'),
                row=2, col=1
            )
        
        # Metrics comparison
        fig.add_trace(
            go.Bar(
                x=['Train Loss', 'Val Loss', 'Train Acc', 'Val Acc'],
                y=[
                    history['train_loss'][-1],
                    history['val_loss'][-1],
                    history['train_acc'][-1],
                    history['val_acc'][-1]
                ],
                name='Final Metrics'
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Training Progress")
        fig.write_html(str(output_dir / "training_progress.html"))
        
        # 2. Individual Plots
        # Loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(output_dir / "loss_plot.png")
        plt.close()
        
        # Accuracy plot
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_acc'], label='Train Acc')
        plt.plot(history['val_acc'], label='Val Acc')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.savefig(output_dir / "accuracy_plot.png")
        plt.close()
        
        if self.use_gdrive:
            self.save_to_gdrive('visualizations/training_progress.html', output_dir / "training_progress.html")
            self.save_to_gdrive('visualizations/loss_plot.png', output_dir / "loss_plot.png")
            self.save_to_gdrive('visualizations/accuracy_plot.png', output_dir / "accuracy_plot.png")
    
    def visualize_model_comparison(self, results: dict, output_dir: Path):
        """
        Create visualizations comparing different models.
        
        Args:
            results (dict): Results from different models
            output_dir (Path): Directory to save visualizations
        """
        # Create comparison plot
        fig = go.Figure()
        
        metrics = ['precision', 'recall', 'f1-score']
        models = list(results.keys())
        
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
        
        # Save comparison plot
        fig.write_html(str(output_dir / "model_comparison.html"))
        
        if self.use_gdrive:
            self.save_to_gdrive('visualizations/model_comparison.html', output_dir / "model_comparison.html")
    
    def visualize_anomaly_distribution(self, data: pd.DataFrame, output_dir: Path):
        """
        Visualize anomaly distribution across different review categories.
        
        Args:
            data (pd.DataFrame): DataFrame containing review data
            output_dir (Path): Directory to save visualizations
        """
        # Create anomaly distribution plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Anomaly Distribution by Rating',
                'Anomaly Distribution by Helpful Votes',
                'Anomaly Distribution by Review Length',
                'Anomaly Distribution by Time'
            )
        )
        
        # 1. By Rating
        rating_dist = data.groupby('rating')['is_helpful'].mean()
        fig.add_trace(
            go.Bar(x=rating_dist.index, y=rating_dist.values, name='By Rating'),
            row=1, col=1
        )
        
        # 2. By Helpful Votes
        helpful_dist = data.groupby('helpful_votes')['is_helpful'].mean()
        fig.add_trace(
            go.Bar(x=helpful_dist.index, y=helpful_dist.values, name='By Helpful Votes'),
            row=1, col=2
        )
        
        # 3. By Review Length
        data['review_length'] = data['text'].str.len()
        length_bins = pd.qcut(data['review_length'], q=10)
        length_dist = data.groupby(length_bins)['is_helpful'].mean()
        fig.add_trace(
            go.Bar(x=length_dist.index.astype(str), y=length_dist.values, name='By Length'),
            row=2, col=1
        )
        
        # 4. By Time
        time_bins = pd.qcut(data['sort_timestamp'], q=10)
        time_dist = data.groupby(time_bins)['is_helpful'].mean()
        fig.add_trace(
            go.Bar(x=time_dist.index.astype(str), y=time_dist.values, name='By Time'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Anomaly Distribution Analysis")
        fig.write_html(str(output_dir / "anomaly_distribution.html"))
        
        if self.use_gdrive:
            self.save_to_gdrive('visualizations/anomaly_distribution.html', output_dir / "anomaly_distribution.html")
    
    def compare_with_baselines(self, train_data: tuple, val_data: tuple, output_dir: Path) -> dict:
        """
        Compare BERT model with baseline models.
        
        Args:
            train_data (tuple): (X_train, y_train, texts_train)
            val_data (tuple): (X_val, y_val, texts_val)
            output_dir (Path): Directory to save results
            
        Returns:
            dict: Comparison results
        """
        X_train, y_train, texts_train = train_data
        X_val, y_val, texts_val = val_data
        
        # Prepare text data for baseline models
        tfidf = self.comparison_models['tfidf']
        X_train_tfidf = tfidf.fit_transform(texts_train)
        X_val_tfidf = tfidf.transform(texts_val)
        
        results = {}
        
        # Train and evaluate each baseline model
        for name, model in self.comparison_models.items():
            if name != 'tfidf':
                self.logger.info(f"Training {name}...")
                model.fit(X_train_tfidf, y_train.cpu().numpy())
                pred = model.predict(X_val_tfidf)
                results[name] = classification_report(y_val.cpu().numpy(), pred, output_dict=True)
        
        # Create comparison visualization
        self.visualize_model_comparison(results, output_dir)
        
        return results

def main():
    """Main function to visualize results."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize visualizer with Google Drive integration
    visualizer = ResultsVisualizer(use_gdrive=True)
    
    try:
        # Find latest checkpoint directory
        checkpoint_dir = Path("models/checkpoints")
        if not checkpoint_dir.exists():
            raise FileNotFoundError("No model checkpoints found")
        
        # Find latest checkpoint
        checkpoints = list(checkpoint_dir.glob("**/training_history.json"))
        if not checkpoints:
            raise FileNotFoundError("No training history found")
        
        latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
        checkpoint_dir = latest_checkpoint.parent
        
        # Load training history
        with open(latest_checkpoint, 'r') as f:
            history = json.load(f)
        
        # Create visualizations
        visualizer.visualize_training_progress(history, checkpoint_dir)
        
        # Load test data for anomaly distribution analysis
        from ..data.amazon_loader import AmazonBooksLoader
        loader = AmazonBooksLoader("data/raw/Books_5.csv")
        test_df = loader.load_data(max_items=1000)
        
        # Create anomaly distribution visualizations
        visualizer.visualize_anomaly_distribution(test_df, checkpoint_dir)
        
        # Compare with baseline models
        X_test, y_test, texts = loader.prepare_data(test_df)
        results = visualizer.compare_with_baselines(
            (X_test, y_test, texts),
            (X_test, y_test, texts),
            checkpoint_dir
        )
        
        # Save comparison results
        results_path = checkpoint_dir / "model_comparison_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        if visualizer.use_gdrive:
            visualizer.save_to_gdrive('results/model_comparison_results.json', results)
        
        print("\nVisualization complete. Results saved to:")
        print(f"Local: {checkpoint_dir}")
        if visualizer.use_gdrive:
            print(f"Google Drive: {visualizer.gdrive_dir}")
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 