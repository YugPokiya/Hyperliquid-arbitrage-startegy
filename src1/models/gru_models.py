import torch
import torch.nn as nn
from typing import Optional, Tuple

class GRUModel(nn.Module):
    """
    A configurable Gated Recurrent Unit (GRU) model in PyTorch.
    
    This model processes a sequence of inputs and produces a single output vector 
    based on the final hidden state (Many-to-One architecture). It is suitable 
    for tasks like time-series forecasting, sentiment analysis, or sequence classification.

    Attributes:
        input_dim (int): Number of features in the input sequence (e.g., 1 for univariate time series).
        hidden_dim (int): Number of features in the hidden state.
        layer_dim (int): Number of stacked GRU layers.
        output_dim (int): Number of output neurons (e.g., 1 for regression, N for classification).
        dropout_prob (float): Dropout probability for regularization (default: 0.0).
    """

    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        layer_dim: int, 
        output_dim: int, 
        dropout_prob: float = 0.0
    ):
        """
        Initializes the GRU Model.

        Args:
            input_dim (int): The number of expected features in the input x.
            hidden_dim (int): The number of features in the hidden state h.
            layer_dim (int): Number of recurrent layers.
            output_dim (int): Dimension of the output (e.g., number of classes).
            dropout_prob (float): If non-zero, introduces a Dropout layer on the outputs 
                                  of each GRU layer except the last layer.
        """
        super(GRUModel, self).__init__()

        # Model hyperparameters
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # GRU Layer
        # batch_first=True causes input/output tensors to be of shape:
        # (batch_size, seq_length, features)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layer_dim,
            batch_first=True,
            dropout=dropout_prob if layer_dim > 1 else 0.0
        )

        # Fully Connected Layer (Readout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        # Initialize hidden state with zeros
        # Shape: (layer_dim, batch_size, hidden_dim)
        # We use x.size(0) to get the current batch size dynamically
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)

        # Forward propagate GRU
        # out shape: (batch_size, seq_length, hidden_dim)
        # hn shape:  (layer_dim, batch_size, hidden_dim)
        out, _ = self.gru(x, h0)

        # Decode the hidden state of the last time step
        # We take the data from the last sequence step: out[:, -1, :]
        out = self.fc(out[:, -1, :])
        
        return out
