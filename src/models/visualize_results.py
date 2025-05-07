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
import os
warnings.filterwarnings('ignore')

class ResultsVisualizer:
    def __init__(self, model_dir: str = "models/checkpoints", use_gdrive: bool = False):
        """
        Initialize the results visualizer.
        
        Args:
            model_dir (str): Directory containing model checkpoints
            use_gdrive (bool): Whether to use Google Drive for storage
        """
        self.model_dir = Path(model_dir)
        self.logger = logging.getLogger(__name__)
        self.use_gdrive = use_gdrive
        
        if use_gdrive:
            self._setup_gdrive()
    
    def _setup_gdrive(self):
        """Set up Google Drive integration."""
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            self.gdrive_dir = Path('/content/drive/MyDrive/amazon_reviews_model')
            self.gdrive_dir.mkdir(parents=True, exist_ok=True)
        except ImportError:
            self.logger.warning("Google Drive integration not available (not running in Colab)")
            self.use_gdrive = False
        except Exception as e:
            self.logger.warning(f"Failed to set up Google Drive: {str(e)}")
            self.use_gdrive = False
    
    def _find_latest_checkpoint(self) -> Path:
        """
        Find the latest checkpoint directory.
        
        Returns:
            Path: Path to the latest checkpoint directory
        """
        try:
            checkpoint_dirs = [d for d in self.model_dir.iterdir() if d.is_dir()]
            if not checkpoint_dirs:
                raise FileNotFoundError("No checkpoint directories found")
            return max(checkpoint_dirs, key=lambda x: x.stat().st_mtime)
        except Exception as e:
            self.logger.error(f"Error finding latest checkpoint: {str(e)}")
            raise
    
    def visualize_training_progress(self):
        """Visualize training progress from the latest checkpoint."""
        try:
            checkpoint_dir = self._find_latest_checkpoint()
            history_path = checkpoint_dir / "training_history.json"
            
            if not history_path.exists():
                raise FileNotFoundError(f"Training history not found at {history_path}")
            
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            # Create output directory
            output_dir = checkpoint_dir / "visualizations"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Plot training metrics
            plt.figure(figsize=(12, 5))
            
            # Plot loss
            plt.subplot(1, 2, 1)
            plt.plot(history['train_loss'], label='Training Loss')
            if 'val_loss' in history:
                plt.plot(history['val_loss'], label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # Plot accuracy
            plt.subplot(1, 2, 2)
            plt.plot(history['train_acc'], label='Training Accuracy')
            if 'val_acc' in history:
                plt.plot(history['val_acc'], label='Validation Accuracy')
            plt.title('Training and Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.tight_layout()
            
            # Save plot
            plot_path = output_dir / "training_progress.png"
            plt.savefig(plot_path)
            plt.close()
            
            # Save to Google Drive if available
            if self.use_gdrive:
                try:
                    gdrive_plot_path = self.gdrive_dir / "visualizations" / "training_progress.png"
                    gdrive_plot_path.parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(gdrive_plot_path)
                except Exception as e:
                    self.logger.warning(f"Failed to save plot to Google Drive: {str(e)}")
            
            self.logger.info(f"Training progress visualization saved to {plot_path}")
            
        except Exception as e:
            self.logger.error(f"Error visualizing training progress: {str(e)}")
            raise
    
    def visualize_model_comparison(self):
        """Visualize comparison of different model checkpoints."""
        try:
            checkpoint_dirs = [d for d in self.model_dir.iterdir() if d.is_dir()]
            if not checkpoint_dirs:
                raise FileNotFoundError("No checkpoint directories found")
            
            # Create output directory
            output_dir = self.model_dir / "visualizations"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Collect metrics from each checkpoint
            metrics = []
            for checkpoint_dir in checkpoint_dirs:
                history_path = checkpoint_dir / "training_history.json"
                if history_path.exists():
                    with open(history_path, 'r') as f:
                        history = json.load(f)
                        metrics.append({
                            'checkpoint': checkpoint_dir.name,
                            'final_loss': history['train_loss'][-1],
                            'final_acc': history['train_acc'][-1]
                        })
            
            if not metrics:
                raise ValueError("No valid metrics found in checkpoints")
            
            # Create comparison plot
            plt.figure(figsize=(12, 5))
            
            # Plot loss comparison
            plt.subplot(1, 2, 1)
            plt.bar([m['checkpoint'] for m in metrics], [m['final_loss'] for m in metrics])
            plt.title('Final Loss by Checkpoint')
            plt.xlabel('Checkpoint')
            plt.ylabel('Loss')
            plt.xticks(rotation=45)
            
            # Plot accuracy comparison
            plt.subplot(1, 2, 2)
            plt.bar([m['checkpoint'] for m in metrics], [m['final_acc'] for m in metrics])
            plt.title('Final Accuracy by Checkpoint')
            plt.xlabel('Checkpoint')
            plt.ylabel('Accuracy')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = output_dir / "model_comparison.png"
            plt.savefig(plot_path)
            plt.close()
            
            # Save to Google Drive if available
            if self.use_gdrive:
                try:
                    gdrive_plot_path = self.gdrive_dir / "visualizations" / "model_comparison.png"
                    gdrive_plot_path.parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(gdrive_plot_path)
                except Exception as e:
                    self.logger.warning(f"Failed to save plot to Google Drive: {str(e)}")
            
            self.logger.info(f"Model comparison visualization saved to {plot_path}")
            
        except Exception as e:
            self.logger.error(f"Error visualizing model comparison: {str(e)}")
            raise
    
    def visualize_evaluation_results(self):
        """Visualize evaluation results from the latest checkpoint."""
        try:
            checkpoint_dir = self._find_latest_checkpoint()
            results_path = checkpoint_dir / "evaluation_results.json"
            
            if not results_path.exists():
                raise FileNotFoundError(f"Evaluation results not found at {results_path}")
            
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            # Create output directory
            output_dir = checkpoint_dir / "visualizations"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            
            # Save confusion matrix plot
            cm_path = output_dir / "confusion_matrix.png"
            plt.savefig(cm_path)
            plt.close()
            
            # Plot ROC curve if probabilities are available
            if 'probabilities' in results:
                plt.figure(figsize=(8, 6))
                fpr, tpr, _ = roc_curve(results['true_labels'], results['probabilities'])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc="lower right")
                
                # Save ROC curve plot
                roc_path = output_dir / "roc_curve.png"
                plt.savefig(roc_path)
                plt.close()
            
            # Save to Google Drive if available
            if self.use_gdrive:
                try:
                    gdrive_output_dir = self.gdrive_dir / "visualizations"
                    gdrive_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Copy confusion matrix
                    gdrive_cm_path = gdrive_output_dir / "confusion_matrix.png"
                    plt.savefig(gdrive_cm_path)
                    
                    # Copy ROC curve if available
                    if 'probabilities' in results:
                        gdrive_roc_path = gdrive_output_dir / "roc_curve.png"
                        plt.savefig(gdrive_roc_path)
                except Exception as e:
                    self.logger.warning(f"Failed to save plots to Google Drive: {str(e)}")
            
            self.logger.info(f"Evaluation visualizations saved to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error visualizing evaluation results: {str(e)}")
            raise
    
    def analyze_book_reviews(self, file_path: str, output_dir: str = None):
        """
        Analyze book reviews from the dataset.
        
        Args:
            file_path (str): Path to the Books_rating.csv file
            output_dir (str): Directory to save visualizations
        """
        try:
            # Load data
            df = pd.read_csv(file_path)
            
            # Create output directory
            if output_dir is None:
                output_dir = self.model_dir / "visualizations"
            else:
                output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Rating Distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x='rating', bins=20)
            plt.title('Distribution of Book Ratings')
            plt.xlabel('Rating')
            plt.ylabel('Count')
            plt.savefig(output_dir / "rating_distribution.png")
            plt.close()
            
            # 2. Rating by Category
            plt.figure(figsize=(12, 6))
            category_ratings = df.groupby('categories')['rating'].mean().sort_values(ascending=False)
            sns.barplot(x=category_ratings.index, y=category_ratings.values)
            plt.title('Average Rating by Category')
            plt.xlabel('Category')
            plt.ylabel('Average Rating')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / "category_ratings.png")
            plt.close()
            
            # 3. Review Length Analysis
            df['review_length'] = df['text'].str.len()
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df, x='review_length', y='rating')
            plt.title('Review Length vs Rating')
            plt.xlabel('Review Length (characters)')
            plt.ylabel('Rating')
            plt.savefig(output_dir / "review_length_vs_rating.png")
            plt.close()
            
            # 4. Create interactive dashboard
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Rating Distribution',
                    'Average Rating by Category',
                    'Review Length vs Rating',
                    'Rating Statistics'
                )
            )
            
            # Add rating distribution
            fig.add_trace(
                go.Histogram(x=df['rating'], name='Rating Distribution'),
                row=1, col=1
            )
            
            # Add category ratings
            fig.add_trace(
                go.Bar(x=category_ratings.index, y=category_ratings.values, name='Category Ratings'),
                row=1, col=2
            )
            
            # Add review length scatter
            fig.add_trace(
                go.Scatter(x=df['review_length'], y=df['rating'], mode='markers', name='Review Length'),
                row=2, col=1
            )
            
            # Add rating statistics
            fig.add_trace(
                go.Box(y=df['rating'], name='Rating Statistics'),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                height=800,
                width=1200,
                title_text="Book Reviews Analysis Dashboard",
                showlegend=False
            )
            
            # Save dashboard
            fig.write_html(output_dir / "book_reviews_dashboard.html")
            
            # 5. Save summary statistics
            summary = {
                'total_reviews': len(df),
                'average_rating': df['rating'].mean(),
                'rating_std': df['rating'].std(),
                'min_rating': df['rating'].min(),
                'max_rating': df['rating'].max(),
                'average_review_length': df['review_length'].mean(),
                'total_categories': len(df['categories'].unique()),
                'top_categories': category_ratings.head(10).to_dict()
            }
            
            with open(output_dir / "review_summary.json", 'w') as f:
                json.dump(summary, f, indent=4)
            
            # Save to Google Drive if available
            if self.use_gdrive:
                try:
                    gdrive_output_dir = self.gdrive_dir / "visualizations"
                    gdrive_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Copy all visualizations
                    for file in output_dir.glob("*"):
                        if file.is_file():
                            gdrive_file = gdrive_output_dir / file.name
                            if file.suffix == '.html':
                                file.write_text(file.read_text())
                            else:
                                plt.savefig(gdrive_file)
                except Exception as e:
                    self.logger.warning(f"Failed to save visualizations to Google Drive: {str(e)}")
            
            self.logger.info(f"Book reviews analysis saved to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error analyzing book reviews: {str(e)}")
            raise

def main():
    """Main function to run visualizations."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize visualizer
    visualizer = ResultsVisualizer(use_gdrive=True)
    
    try:
        # Visualize training progress
        visualizer.visualize_training_progress()
        
        # Visualize model comparison
        visualizer.visualize_model_comparison()
        
        # Visualize evaluation results
        visualizer.visualize_evaluation_results()
        
        # Analyze book reviews
        visualizer.analyze_book_reviews("data/raw/Books_rating.csv")
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 