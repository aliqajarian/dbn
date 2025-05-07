import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from pathlib import Path
import logging
from train_amazon_optimized import TimeSeriesDBN, AmazonBooksLoader, OptimizedAmazonTrainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model_path: str = "models/amazon_anomaly_detector.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        
    def load_model(self, model_path: str) -> TimeSeriesDBN:
        """Load the trained model."""
        model = TimeSeriesDBN(layer_sizes=[3, 64, 32, 3])
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def calculate_reconstruction_error(self, data: torch.Tensor) -> np.ndarray:
        """Calculate reconstruction error for each sample."""
        with torch.no_grad():
            data = data.to(self.device)
            reconstruction = self.model.reconstruct(data)
            errors = torch.mean((data - reconstruction) ** 2, dim=1)
            return errors.cpu().numpy()
    
    def detect_anomalies(self, errors: np.ndarray, threshold: float = None) -> np.ndarray:
        """Detect anomalies using reconstruction error."""
        if threshold is None:
            threshold = np.percentile(errors, 95)  # 5% of data as anomalies
        return errors > threshold
    
    def evaluate_performance(self, errors: np.ndarray, true_labels: np.ndarray = None) -> dict:
        """Calculate performance metrics."""
        metrics = {}
        
        # Basic statistics
        metrics['mean_error'] = np.mean(errors)
        metrics['std_error'] = np.std(errors)
        metrics['max_error'] = np.max(errors)
        metrics['min_error'] = np.min(errors)
        
        # If true labels are available, calculate additional metrics
        if true_labels is not None:
            predictions = self.detect_anomalies(errors)
            metrics['roc_auc'] = roc_auc_score(true_labels, errors)
            metrics['avg_precision'] = average_precision_score(true_labels, errors)
            
            # Calculate precision-recall curve
            precision, recall, thresholds = precision_recall_curve(true_labels, errors)
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['thresholds'] = thresholds
        
        return metrics
    
    def visualize_results(self, errors: np.ndarray, true_labels: np.ndarray = None):
        """Create visualizations of the results."""
        # Create output directory
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        
        # 1. Reconstruction Error Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, bins=50)
        plt.title('Distribution of Reconstruction Errors')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Count')
        plt.savefig(output_dir / 'error_distribution.png')
        plt.close()
        
        # 2. Error vs Sample Index
        plt.figure(figsize=(12, 6))
        plt.plot(errors)
        plt.title('Reconstruction Error by Sample')
        plt.xlabel('Sample Index')
        plt.ylabel('Reconstruction Error')
        plt.savefig(output_dir / 'error_by_sample.png')
        plt.close()
        
        if true_labels is not None:
            # 3. ROC Curve
            plt.figure(figsize=(8, 8))
            fpr, tpr, _ = roc_curve(true_labels, errors)
            plt.plot(fpr, tpr)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.title('ROC Curve')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.savefig(output_dir / 'roc_curve.png')
            plt.close()
            
            # 4. Precision-Recall Curve
            plt.figure(figsize=(8, 8))
            precision, recall, _ = precision_recall_curve(true_labels, errors)
            plt.plot(recall, precision)
            plt.title('Precision-Recall Curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.savefig(output_dir / 'precision_recall_curve.png')
            plt.close()
    
    def generate_paper_results(self, errors: np.ndarray, true_labels: np.ndarray = None) -> pd.DataFrame:
        """Generate results suitable for a paper."""
        results = []
        
        # Calculate metrics at different thresholds
        thresholds = np.percentile(errors, [90, 95, 97.5, 99])
        
        for threshold in thresholds:
            predictions = errors > threshold
            if true_labels is not None:
                precision = precision_score(true_labels, predictions)
                recall = recall_score(true_labels, predictions)
                f1 = f1_score(true_labels, predictions)
            else:
                precision = recall = f1 = np.nan
                
            results.append({
                'Threshold': threshold,
                'Anomaly_Rate': np.mean(predictions),
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1
            })
        
        return pd.DataFrame(results)

def main():
    # Load test data
    loader = AmazonBooksLoader()
    test_data = loader.load_data(max_records=5000)  # Use different data than training
    
    if test_data is None:
        logger.error("Failed to load test data")
        return
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Prepare data
    trainer = OptimizedAmazonTrainer()
    test_data = trainer.prepare_data(test_data)
    features = trainer.create_features(test_data)
    test_tensor = torch.FloatTensor(features)
    
    # Calculate reconstruction errors
    errors = evaluator.calculate_reconstruction_error(test_tensor)
    
    # Evaluate performance
    metrics = evaluator.evaluate_performance(errors)
    
    # Log metrics
    logger.info("\nPerformance Metrics:")
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"{metric}: {value:.4f}")
    
    # Visualize results
    evaluator.visualize_results(errors)
    
    # Generate paper results
    paper_results = evaluator.generate_paper_results(errors)
    paper_results.to_csv('results/paper_metrics.csv', index=False)
    logger.info("\nPaper results saved to results/paper_metrics.csv")
    
    # Save detailed results
    results_df = pd.DataFrame({
        'reconstruction_error': errors,
        'is_anomaly': evaluator.detect_anomalies(errors)
    })
    results_df.to_csv('results/detailed_results.csv', index=False)
    logger.info("Detailed results saved to results/detailed_results.csv")

if __name__ == "__main__":
    main() 