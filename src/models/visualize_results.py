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
    
    def visualize_training_progress(self, history: dict, output_dir: Path):
        """
        Create visualizations for training progress.
        
        Args:
            history (dict): Training history containing loss and accuracy metrics
            output_dir (Path): Directory to save visualizations
        """
        try:
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Loss Plot
            plt.figure(figsize=(10, 6))
            plt.plot(history['train_loss'], label='Training Loss')
            if 'val_loss' in history:
                plt.plot(history['val_loss'], label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(output_dir / "loss_plot.png")
            plt.close()
            
            # 2. Accuracy Plot
            plt.figure(figsize=(10, 6))
            plt.plot(history['train_acc'], label='Training Accuracy')
            if 'val_acc' in history:
                plt.plot(history['val_acc'], label='Validation Accuracy')
            plt.title('Training and Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig(output_dir / "accuracy_plot.png")
            plt.close()
            
            # 3. Interactive Dashboard
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Loss', 'Accuracy')
            )
            
            # Add loss traces
            fig.add_trace(
                go.Scatter(y=history['train_loss'], name='Training Loss'),
                row=1, col=1
            )
            if 'val_loss' in history:
                fig.add_trace(
                    go.Scatter(y=history['val_loss'], name='Validation Loss'),
                    row=1, col=1
                )
            
            # Add accuracy traces
            fig.add_trace(
                go.Scatter(y=history['train_acc'], name='Training Accuracy'),
                row=2, col=1
            )
            if 'val_acc' in history:
                fig.add_trace(
                    go.Scatter(y=history['val_acc'], name='Validation Accuracy'),
                    row=2, col=1
                )
            
            fig.update_layout(height=800, title_text="Training Progress Dashboard")
            fig.write_html(str(output_dir / "training_dashboard.html"))
            
            # Save to Google Drive if enabled
            if self.use_gdrive:
                try:
                    gdrive_viz_dir = self.gdrive_dir / "visualizations" / "training"
                    gdrive_viz_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Copy visualization files
                    for file in ['loss_plot.png', 'accuracy_plot.png', 'training_dashboard.html']:
                        src_path = output_dir / file
                        dst_path = gdrive_viz_dir / file
                        if src_path.exists():
                            import shutil
                            shutil.copy2(src_path, dst_path)
                except Exception as e:
                    self.logger.warning(f"Failed to save visualizations to Google Drive: {str(e)}")
            
            self.logger.info(f"Saved training visualizations to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error creating training visualizations: {str(e)}")
            raise
    
    def visualize_model_comparison(self, results: dict, output_dir: Path):
        """
        Create visualizations comparing different models.
        
        Args:
            results (dict): Dictionary containing results for different models
            output_dir (Path): Directory to save visualizations
        """
        try:
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
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
            fig.write_html(str(output_dir / "model_comparison.html"))
            
            # Save to Google Drive if enabled
            if self.use_gdrive:
                try:
                    gdrive_viz_dir = self.gdrive_dir / "visualizations" / "comparison"
                    gdrive_viz_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Copy visualization file
                    src_path = output_dir / "model_comparison.html"
                    dst_path = gdrive_viz_dir / "model_comparison.html"
                    if src_path.exists():
                        import shutil
                        shutil.copy2(src_path, dst_path)
                except Exception as e:
                    self.logger.warning(f"Failed to save comparison visualization to Google Drive: {str(e)}")
            
            self.logger.info(f"Saved model comparison visualization to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error creating model comparison visualization: {str(e)}")
            raise
    
    def visualize_evaluation_results(self, labels: list, predictions: list, probabilities: list, output_dir: Path):
        """
        Create visualizations for evaluation results.
        
        Args:
            labels (list): True labels
            predictions (list): Model predictions
            probabilities (list): Prediction probabilities
            output_dir (Path): Directory to save visualizations
        """
        try:
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Confusion Matrix
            conf_matrix = confusion_matrix(labels, predictions)
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(output_dir / "confusion_matrix.png")
            plt.close()
            
            # 2. ROC Curve
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
            plt.savefig(output_dir / "roc_curve.png")
            plt.close()
            
            # 3. Interactive Dashboard
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
            fig.write_html(str(output_dir / "evaluation_dashboard.html"))
            
            # Save to Google Drive if enabled
            if self.use_gdrive:
                try:
                    gdrive_viz_dir = self.gdrive_dir / "visualizations" / "evaluation"
                    gdrive_viz_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Copy visualization files
                    for file in ['confusion_matrix.png', 'roc_curve.png', 'evaluation_dashboard.html']:
                        src_path = output_dir / file
                        dst_path = gdrive_viz_dir / file
                        if src_path.exists():
                            import shutil
                            shutil.copy2(src_path, dst_path)
                except Exception as e:
                    self.logger.warning(f"Failed to save evaluation visualizations to Google Drive: {str(e)}")
            
            self.logger.info(f"Saved evaluation visualizations to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error creating evaluation visualizations: {str(e)}")
            raise

def main():
    """Main function to create visualizations."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize visualizer with Google Drive integration
    visualizer = ResultsVisualizer(use_gdrive=True)
    
    try:
        # Find latest checkpoint directory
        checkpoint_dir = Path("models/checkpoints")
        if not checkpoint_dir.exists():
            raise FileNotFoundError("No model checkpoints found")
        
        # Find latest checkpoint directory
        checkpoint_dirs = [d for d in checkpoint_dir.iterdir() if d.is_dir()]
        if not checkpoint_dirs:
            raise FileNotFoundError("No checkpoint directories found")
        
        latest_dir = max(checkpoint_dirs, key=lambda x: x.stat().st_mtime)
        
        # Load training history
        history_path = latest_dir / "training_history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            # Create training visualizations
            visualizer.visualize_training_progress(history, latest_dir)
        
        # Load evaluation results
        results_path = latest_dir / "evaluation_results.json"
        if results_path.exists():
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            # Create evaluation visualizations
            if 'labels' in results and 'predictions' in results and 'probabilities' in results:
                visualizer.visualize_evaluation_results(
                    results['labels'],
                    results['predictions'],
                    results['probabilities'],
                    latest_dir
                )
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 