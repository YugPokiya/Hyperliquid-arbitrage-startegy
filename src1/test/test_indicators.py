import pytest
import torch
import numpy as np
from utils.indicators import calculate_fisher_transform

def test_fisher_transform_shape():
    """Ensure output shape matches input shape."""
    length = 100
    high = torch.rand(length)
    low = torch.rand(length)
    
    fisher = calculate_fisher_transform(high, low, length=9)
    assert fisher.shape == (length,)

def test_fisher_transform_values():
    """
    Test logic on controlled data.
    If price is constantly rising, Fisher should generally be positive.
    """
    # Create a linear trend up
    price = torch.linspace(10, 20, steps=50)
    high = price + 0.5
    low = price - 0.5
    
    fisher = calculate_fisher_transform(high, low, length=9)
    
    # Check the last few values are positive (uptrend)
    assert torch.all(fisher[-5:] > 0)

def test_fisher_transform_clamping():
    """Ensure the function handles extreme inputs without returning NaNs."""
    # Flat line data (denom becomes 0)
    high = torch.full((50,), 10.0)
    low = torch.full((50,), 10.0)
    
    fisher = calculate_fisher_transform(high, low, length=9)
    
    # Should not contain NaNs
    assert not torch.isnan(fisher).any()
    # Should be close to zero for flat line
    assert torch.abs(fisher[-1]) < 0.1
