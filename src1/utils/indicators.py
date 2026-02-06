import torch

def calculate_fisher_transform(
    high: torch.Tensor, 
    low: torch.Tensor, 
    length: int = 9, 
    device: torch.device = None
) -> torch.Tensor:
    """
    Calculates the Fisher Transform using PyTorch.
    
    Formula:
    1. Midpoint = (High + Low) / 2
    2. Normalize price to range [-1, 1] over 'length' window.
    3. Fisher = 0.5 * ln((1 + X) / (1 - X)) + 0.5 * Fisher[prev]
    
    Args:
        high (torch.Tensor): Tensor of High prices.
        low (torch.Tensor): Tensor of Low prices.
        length (int): Lookback period for normalization.
        device (torch.device): 'cpu' or 'cuda'.
        
    Returns:
        torch.Tensor: The Fisher Transform values.
    """
    if device is None:
        device = high.device

    # Ensure inputs are on the correct device
    high = high.to(device)
    low = low.to(device)
    
    n = len(high)
    hl2 = (high + low) / 2.0
    
    # 1. Rolling Min/Max Implementation using Unfold (Vectorized-ish)
    # We pad the beginning to maintain sequence length
    # Note: For massive datasets, you might use 1d pooling, but loops are often fine for 1D logic
    # Here is a loop-based rolling min/max for clarity and memory safety on time-series
    
    highest_high = torch.zeros_like(high)
    lowest_low = torch.zeros_like(low)
    
    # Simple sliding window for min/max
    for i in range(n):
        start_idx = max(0, i - length + 1)
        highest_high[i] = torch.max(high[start_idx : i + 1])
        lowest_low[i] = torch.min(low[start_idx : i + 1])

    # 2. Normalize to [-1, 1] (Stochastics-like calculation)
    # Value1 = 0.66 * ((HL2 - Low) / (High - Low) - 0.5) + 0.67 * Value1_prev
    
    value1 = torch.zeros_like(hl2)
    fisher = torch.zeros_like(hl2)
    
    eps = 1e-9 # Epsilon to prevent division by zero
    
    # Recursive calculation loop
    # (Recursive filters are hard to vectorize perfectly without compiling)
    for i in range(1, n):
        denom = highest_high[i] - lowest_low[i]
        if denom == 0:
            denom = eps
            
        # Normalize to -0.5 to 0.5 range
        norm_price = (hl2[i] - lowest_low[i]) / denom - 0.5
        
        # Smoothed value (Value1)
        val = 0.66 * norm_price + 0.67 * value1[i-1]
        
        # Clamp to prevent log(0) or log(neg)
        val = torch.clamp(val, -0.999, 0.999)
        value1[i] = val
        
        # Fisher calculation
        fisher[i] = 0.5 * torch.log((1 + val) / (1 - val)) + 0.5 * fisher[i-1]
        
    return fisher
