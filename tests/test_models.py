import pytest
import torch
from src.models.dbn import TimeSeriesDBN

def test_dbn_initialization():
    layer_sizes = [1000, 500, 200, 100]
    model = TimeSeriesDBN(layer_sizes)
    assert isinstance(model, TimeSeriesDBN)
    assert len(model.layer_sizes) == len(layer_sizes)

def test_dbn_forward():
    layer_sizes = [1000, 500, 200, 100]
    model = TimeSeriesDBN(layer_sizes)
    batch_size = 32
    sequence_length = 10
    x = torch.randn(batch_size, sequence_length, layer_sizes[0])
    output = model(x)
    assert output.shape == (batch_size, 1) 