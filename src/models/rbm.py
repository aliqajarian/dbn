import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden, k=1):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.1)
        self.visible_bias = nn.Parameter(torch.zeros(n_visible))
        self.hidden_bias = nn.Parameter(torch.zeros(n_hidden))
        self.k = k

    def sample_hidden(self, visible):
        hidden_prob = torch.sigmoid(torch.matmul(visible, self.W) + self.hidden_bias)
        hidden_sample = torch.bernoulli(hidden_prob)
        return hidden_prob, hidden_sample

    def sample_visible(self, hidden):
        visible_prob = torch.sigmoid(torch.matmul(hidden, self.W.t()) + self.visible_bias)
        visible_sample = torch.bernoulli(visible_prob)
        return visible_prob, visible_sample

    def contrastive_divergence(self, v0):
        h0_prob, h0_sample = self.sample_hidden(v0)
        
        # Gibbs sampling
        vk = v0
        hk = h0_sample
        
        for _ in range(self.k):
            vk_prob, vk = self.sample_visible(hk)
            hk_prob, hk = self.sample_hidden(vk)
        
        return v0, h0_prob, vk, hk_prob

    def forward(self, v):
        h_prob, _ = self.sample_hidden(v)
        return h_prob 