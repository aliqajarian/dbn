import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.stats import wasserstein_distance
from typing import Dict, Tuple

class GANMetrics:
    @staticmethod
    def compute_feature_statistics(data: torch.Tensor) -> Dict[str, np.ndarray]:
        """Compute basic statistics for each feature."""
        return {
            'mean': data.mean(dim=0).cpu().numpy(),
            'std': data.std(dim=0).cpu().numpy(),
            'min': data.min(dim=0)[0].cpu().numpy(),
            'max': data.max(dim=0)[0].cpu().numpy()
        }
    
    @staticmethod
    def compute_wasserstein_distance(real_data: torch.Tensor, 
                                   fake_data: torch.Tensor) -> float:
        """Compute Wasserstein distance between real and generated data."""
        real_np = real_data.cpu().numpy()
        fake_np = fake_data.cpu().numpy()
        
        # Compute for each feature
        distances = []
        for i in range(real_np.shape[1]):
            distances.append(wasserstein_distance(real_np[:, i], fake_np[:, i]))
        
        return np.mean(distances)
    
    @staticmethod
    def compute_diversity_score(fake_data: torch.Tensor) -> float:
        """Compute diversity score of generated data."""
        # Compute pairwise distances
        distances = torch.cdist(fake_data, fake_data)
        # Remove self-distances
        distances = distances[~torch.eye(distances.shape[0], dtype=bool)]
        return distances.mean().item()
    
    @staticmethod
    def evaluate_gan_performance(real_data: torch.Tensor,
                               fake_data: torch.Tensor,
                               discriminator: torch.nn.Module) -> Dict[str, float]:
        """Compute comprehensive GAN evaluation metrics."""
        # Basic statistics comparison
        real_stats = GANMetrics.compute_feature_statistics(real_data)
        fake_stats = GANMetrics.compute_feature_statistics(fake_data)
        
        # Compute Wasserstein distance
        w_distance = GANMetrics.compute_wasserstein_distance(real_data, fake_data)
        
        # Compute diversity score
        diversity = GANMetrics.compute_diversity_score(fake_data)
        
        # Get discriminator scores
        with torch.no_grad():
            real_scores = discriminator(real_data)
            fake_scores = discriminator(fake_data)
        
        # Compute precision, recall, and F1 for discriminator
        real_preds = (real_scores > 0.5).float()
        fake_preds = (fake_scores > 0.5).float()
        
        precision = precision_score(
            [1] * len(real_preds) + [0] * len(fake_preds),
            torch.cat([real_preds, fake_preds]).cpu().numpy()
        )
        
        recall = recall_score(
            [1] * len(real_preds) + [0] * len(fake_preds),
            torch.cat([real_preds, fake_preds]).cpu().numpy()
        )
        
        f1 = f1_score(
            [1] * len(real_preds) + [0] * len(fake_preds),
            torch.cat([real_preds, fake_preds]).cpu().numpy()
        )
        
        return {
            'wasserstein_distance': w_distance,
            'diversity_score': diversity,
            'discriminator_precision': precision,
            'discriminator_recall': recall,
            'discriminator_f1': f1,
            'real_mean_score': real_scores.mean().item(),
            'fake_mean_score': fake_scores.mean().item()
        } 