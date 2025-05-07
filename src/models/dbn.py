import torch
import torch.nn as nn
from typing import List, Tuple
import numpy as np
from .rbm import RBM

class TimeSeriesDBN(nn.Module):
    def __init__(self, 
                 layer_sizes: List[int],
                 sequence_length: int = 10,
                 k: int = 1):
        super(TimeSeriesDBN, self).__init__()
        self.layer_sizes = layer_sizes
        self.sequence_length = sequence_length
        
        # Create RBMs
        self.rbms = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            rbm = RBM(layer_sizes[i], layer_sizes[i + 1], k=k)
            self.rbms.append(rbm)
        
        # LSTM layer for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=layer_sizes[-1],
            hidden_size=layer_sizes[-1],
            num_layers=2,
            batch_first=True
        )
        
        # Final classification layer
        self.classifier = nn.Linear(layer_sizes[-1], 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Process through RBMs
        h = x
        for rbm in self.rbms:
            h = rbm(h)
        
        # Reshape for LSTM
        h = h.view(batch_size, self.sequence_length, -1)
        
        # Process through LSTM
        lstm_out, _ = self.lstm(h)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Final classification
        return torch.sigmoid(self.classifier(last_output))
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input through the network."""
        h = x
        for rbm in self.rbms:
            h = rbm(h)
        
        # Reconstruct back through the network
        v = h
        for rbm in reversed(self.rbms):
            v = rbm.sample_visible(v)[0]
        
        return v
    
    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate reconstruction error."""
        reconstructed = self.reconstruct(x)
        return torch.mean((x - reconstructed) ** 2, dim=1)
    
    def detect_anomalies(self, 
                        x: torch.Tensor, 
                        threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Detect anomalies using reconstruction error."""
        errors = self.get_reconstruction_error(x)
        anomalies = errors > threshold
        return anomalies, errors 