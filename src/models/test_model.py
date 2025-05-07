import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from ..utils.review_visualizer import ReviewVisualizer
from .review_analyzer import ReviewAnalyzer

class ModelTester:
    def __init__(self, model_dir: str = "models/checkpoints"):
        """
        Initialize the model tester.
        
        Args:
            model_dir (str): Directory containing model checkpoints
        """
        self.model_dir = Path(model_dir)
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.visualizer = ReviewVisualizer()
    
    def load_model(self, checkpoint_path: str) -> ReviewAnalyzer:
        """
        Load a trained model from checkpoint.
        
        Args:
            checkpoint_path (str): Path to the model checkpoint
            
        Returns:
            ReviewAnalyzer: Loaded model
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Initialize model
        model = ReviewAnalyzer()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def test(self, 
             model: ReviewAnalyzer,
             test_data: pd.DataFrame,
             batch_size: int = 32) -> dict:
        """
        Test the model on test data.
        
        Args:
            model (ReviewAnalyzer): Trained model
            test_data (pd.DataFrame): Test data
            batch_size (int): Batch size for testing
            
        Returns:
            dict: Test results and metrics
        """
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self._get_batches(test_data, batch_size):
                outputs = model(batch['text'])
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        # Calculate metrics
        results = {
            'predictions': all_predictions,
            'true_labels': all_labels,
            'classification_report': classification_report(all_labels, all_predictions),
            'confusion_matrix': confusion_matrix(all_labels, all_predictions)
        }
        
        return results
    
    def _get_batches(self, data: pd.DataFrame, batch_size: int):
        """Generate batches of data for testing."""
        for i in range(0, len(data), batch_size):
            batch_data = data.iloc[i:i + batch_size]
            yield {
                'text': batch_data['review_text'].values,
                'labels': torch.tensor(batch_data['is_anomaly'].values, 
                                     dtype=torch.long).to(self.device)
            }
    
    def visualize_results(self, results: dict, save_dir: str = None):
        """
        Visualize test results.
        
        Args:
            results (dict): Test results
            save_dir (str): Directory to save visualizations
        """
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(results['confusion_matrix'], 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        if save_dir:
            save_path = Path(save_dir) / 'confusion_matrix.png'
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
        
        # Print classification report
        print("\nClassification Report:")
        print(results['classification_report'])
    
    def analyze_errors(self, test_data: pd.DataFrame, results: dict):
        """
        Analyze misclassified examples.
        
        Args:
            test_data (pd.DataFrame): Test data
            results (dict): Test results
        """
        predictions = np.array(results['predictions'])
        true_labels = np.array(results['true_labels'])
        
        # Find misclassified examples
        misclassified = test_data[predictions != true_labels]
        
        print(f"\nFound {len(misclassified)} misclassified examples")
        
        # Analyze false positives
        false_positives = misclassified[predictions[misclassified.index] == 1]
        print(f"\nFalse Positives (Normal reviews classified as anomalies):")
        for _, review in false_positives.head().iterrows():
            print(f"\nReview: {review['review_text'][:200]}...")
            print(f"True Label: Normal")
            print(f"Predicted: Anomaly")
        
        # Analyze false negatives
        false_negatives = misclassified[predictions[misclassified.index] == 0]
        print(f"\nFalse Negatives (Anomalous reviews classified as normal):")
        for _, review in false_negatives.head().iterrows():
            print(f"\nReview: {review['review_text'][:200]}...")
            print(f"True Label: Anomaly")
            print(f"Predicted: Normal")

def main():
    """Main function to test the model."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize tester
    tester = ModelTester()
    
    # Load test data
    from ..data.amazon_loader import AmazonBooksLoader
    loader = AmazonBooksLoader("data/raw/meta_Books.jsonl.gz")
    
    # Load and prepare test data
    books_df = loader.load_data(max_items=1000)
    reviews_df = loader.get_reviews("data/raw/reviews_Books.jsonl.gz", max_reviews=10000)
    merged_df = loader.merge_books_and_reviews(books_df, reviews_df)
    test_df = loader.prepare_for_analysis(merged_df)
    
    # Load the best model checkpoint
    checkpoint_path = "models/checkpoints/latest/final_model.pt"
    model = tester.load_model(checkpoint_path)
    
    # Test model
    results = tester.test(model, test_df)
    
    # Visualize results
    tester.visualize_results(results, save_dir="models/results")
    
    # Analyze errors
    tester.analyze_errors(test_df, results)

if __name__ == "__main__":
    main() 