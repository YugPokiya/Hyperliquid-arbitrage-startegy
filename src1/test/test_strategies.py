import pytest
import torch
import pandas as pd
from utils.indicators import calculate_fisher_transform

def test_strategy_signal_logic():
    """
    Mock test to simulate how the strategy uses the indicator.
    We are testing:
    1. Fisher calculation
    2. Anomaly detection (Threshold crossing)
    """
    # 1. Setup Data
    length = 50
    high = torch.rand(length) * 100
    low = high - 5
    
    # 2. Run Indicator
    fisher = calculate_fisher_transform(high, low)
    
    # 3. Simulate Signal Logic (Anomaly Threshold)
    THRESHOLD = 2.0
    
    # Create fake massive spike in Fisher to test detection
    fisher[-1] = 2.5 # Force an anomaly
    
    is_anomaly = fisher > THRESHOLD
    
    assert is_anomaly[-1] == True
    assert is_anomaly[-1].item() is True
