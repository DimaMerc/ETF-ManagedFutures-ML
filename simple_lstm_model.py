# Modify simple_lstm_model.py to include additional features and better model architecture

import torch
import torch.nn as nn
import math

class SimpleLSTM(nn.Module):
    """
    Enhanced LSTM model for trend prediction with better signal generation.
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        """
        Initialize the enhanced LSTM model.
        
        Parameters:
        -----------
        input_size : int
            Input feature dimension (default: 1 for univariate time series)
        hidden_size : int
            Hidden state dimension (default: 64 to match saved models)
        num_layers : int
            Number of LSTM layers (default: 2 to match saved models)
        output_size : int
            Output dimension (default: 1 for univariate prediction)
        dropout : float
            Dropout probability (default: 0.2 to match saved models)
        """
        super(SimpleLSTM, self).__init__()
        
        # Store parameters as attributes for later reference
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout_rate = dropout
        
        # Enhanced LSTM with better initialization and regularization
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True  # This doubles the effective hidden size
        )
        
        # Add batch normalization for more stable training
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        
        # Add an attention mechanism
        self.attention_w = nn.Parameter(torch.randn(hidden_size * 2, 1))
        
        # Multiple prediction heads for better performance
        # Direction head focuses on trend direction
        self.direction_head = nn.Linear(hidden_size * 2, 1)
        
        # Magnitude head focuses on trend strength
        self.magnitude_head = nn.Linear(hidden_size * 2, 1)
        
        # Final fully connected layer combines the heads
        self.fc = nn.Linear(hidden_size * 2, output_size)
        
        # Initialize weights properly for better convergence
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with optimized strategy."""
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:  # Input-to-hidden weights
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:  # Hidden-to-hidden weights
                nn.init.orthogonal_(param.data)  # Better for RNNs
            elif 'bias' in name:  # Bias terms
                param.data.fill_(0)
                # Set forget gate bias to 1.0 for better gradient flow
                if param.data.size(0) % 4 == 0:  # LSTM has 4 gates per layer
                    forget_gate_offset = param.data.size(0) // 4
                    param.data[forget_gate_offset:forget_gate_offset*2].fill_(1.0)
        
        # Initialize attention weights
        nn.init.xavier_uniform_(self.attention_w)
        
        # Initialize linear layers
        nn.init.xavier_uniform_(self.direction_head.weight)
        nn.init.zeros_(self.direction_head.bias)
        
        nn.init.xavier_uniform_(self.magnitude_head.weight)
        nn.init.zeros_(self.magnitude_head.bias)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def _apply_attention(self, lstm_out):
        """Apply attention mechanism to LSTM outputs."""
        # lstm_out shape: [batch_size, seq_len, hidden_size*2]
        
        # Calculate attention scores
        attn_scores = torch.matmul(lstm_out, self.attention_w)  # [batch_size, seq_len, 1]
        
        # Apply softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=1)  # [batch_size, seq_len, 1]
        
        # Apply attention weights to get context vector
        context = torch.sum(lstm_out * attn_weights, dim=1)  # [batch_size, hidden_size*2]
        
        return context, attn_weights.squeeze(-1)
    
    def forward(self, x):
        """
        Forward pass of the LSTM model.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape [batch_size, seq_length, input_size]
                
        Returns:
        --------
        tuple
            (output, attention_weights)
        """
        # Pass through LSTM layers
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_size*2]
        
        # Apply attention to focus on important parts of the sequence
        context, attn_weights = self._apply_attention(lstm_out)  # [batch_size, hidden_size*2]
        
        # Apply batch normalization for training stability
        if context.size(0) > 1:  # Only apply when batch size > 1
            context_bn = self.batch_norm(context)
        else:
            context_bn = context
        
        # Get direction and magnitude predictions from separate heads
        direction = torch.tanh(self.direction_head(context_bn))  # Range [-1, 1]
        magnitude = torch.sigmoid(self.magnitude_head(context_bn))  # Range [0, 1]
        
        # Get final prediction from combined network
        # The final output is influenced by both direction and magnitude
        output = self.fc(context_bn)
        
        # Blend with direction and magnitude for better performance
        output = output * 0.4 + (direction * magnitude) * 0.6
        
        return output, attn_weights