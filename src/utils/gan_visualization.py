import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class GANVisualizer:
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.training_history = {
            'g_loss': [],
            'd_loss': [],
            'real_scores': [],
            'fake_scores': []
        }
    
    def update_history(self, g_loss: float, d_loss: float, 
                      real_scores: torch.Tensor, fake_scores: torch.Tensor):
        """Update training history with new metrics."""
        self.training_history['g_loss'].append(g_loss)
        self.training_history['d_loss'].append(d_loss)
        self.training_history['real_scores'].append(real_scores.mean().item())
        self.training_history['fake_scores'].append(fake_scores.mean().item())
    
    def plot_training_progress(self):
        """Plot GAN training progress using Plotly."""
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=('Generator and Discriminator Losses',
                                         'Real vs Fake Scores'))
        
        # Plot losses
        fig.add_trace(
            go.Scatter(y=self.training_history['g_loss'], name='Generator Loss'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(y=self.training_history['d_loss'], name='Discriminator Loss'),
            row=1, col=1
        )
        
        # Plot scores
        fig.add_trace(
            go.Scatter(y=self.training_history['real_scores'], name='Real Scores'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(y=self.training_history['fake_scores'], name='Fake Scores'),
            row=2, col=1
        )
        
        fig.update_layout(height=800, title_text="GAN Training Progress")
        fig.show()
    
    def plot_feature_distributions(self, real_data: torch.Tensor, 
                                 fake_data: torch.Tensor, 
                                 feature_indices: List[int] = None):
        """Plot distributions of selected features for real and generated data."""
        if feature_indices is None:
            feature_indices = range(min(5, real_data.shape[1]))
        
        fig = make_subplots(rows=len(feature_indices), cols=1,
                           subplot_titles=[self.feature_names[i] for i in feature_indices])
        
        for idx, feature_idx in enumerate(feature_indices, 1):
            fig.add_trace(
                go.Histogram(x=real_data[:, feature_idx].cpu().numpy(),
                           name='Real', opacity=0.7),
                row=idx, col=1
            )
            fig.add_trace(
                go.Histogram(x=fake_data[:, feature_idx].cpu().numpy(),
                           name='Generated', opacity=0.7),
                row=idx, col=1
            )
        
        fig.update_layout(height=300*len(feature_indices),
                         title_text="Feature Distributions: Real vs Generated")
        fig.show()
    
    def plot_latent_space(self, real_data: torch.Tensor, 
                         fake_data: torch.Tensor,
                         labels: torch.Tensor = None):
        """Plot 2D projection of real and generated data in latent space."""
        from sklearn.manifold import TSNE
        
        # Combine real and fake data
        combined_data = torch.cat([real_data, fake_data], dim=0)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        embedded_data = tsne.fit_transform(combined_data.cpu().numpy())
        
        # Split back into real and fake
        n_real = real_data.shape[0]
        real_embedded = embedded_data[:n_real]
        fake_embedded = embedded_data[n_real:]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=real_embedded[:, 0], y=real_embedded[:, 1],
            mode='markers', name='Real Data',
            marker=dict(color='blue', size=8)
        ))
        fig.add_trace(go.Scatter(
            x=fake_embedded[:, 0], y=fake_embedded[:, 1],
            mode='markers', name='Generated Data',
            marker=dict(color='red', size=8)
        ))
        
        fig.update_layout(title="t-SNE Visualization of Real vs Generated Data",
                         xaxis_title="t-SNE 1", yaxis_title="t-SNE 2")
        fig.show() 