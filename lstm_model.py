# lstm_model.py - Optimized with RTX 5090 CUDA enhancements

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
import math
import time
import logging

logger = logging.getLogger(__name__)



# Add to lstm_model.py, ideally after the imports but before the class definitions
def custom_return_maximizing_loss(y_pred, y_true):
    """
    Custom loss function that rewards positive returns more than it penalizes volatility.
    
    Parameters:
    -----------
    y_pred : torch.Tensor
        Predicted values
    y_true : torch.Tensor
        True values
        
    Returns:
    --------
    torch.Tensor
        Loss value
    """
    # Import torch if not already in scope
    import torch
    import torch.nn as nn
    
    # Standard MSE loss component
    mse_loss = nn.MSELoss()(y_pred, y_true)
    
    # Return maximization component - reward positive direction accuracy more
    directional_accuracy = torch.mean(torch.sign(y_pred) * torch.sign(y_true))
    
    # Sharpe ratio approximation component
    pred_returns = y_pred.view(-1)
    return_mean = torch.mean(pred_returns)
    return_std = torch.std(pred_returns) + 1e-6  # avoid division by zero
    sharpe_approx = return_mean / return_std
    
    # Combined loss with emphasis on returns
    combined_loss = mse_loss - 0.2 * directional_accuracy - 0.1 * sharpe_approx
    
    return combined_loss


class AttentionModule(nn.Module):
    """
    Enhanced attention mechanism for time series focus optimized for Tensor Cores.
    Uses scaled attention for better numerical stability and Tensor Core optimization.
    """
    
    def __init__(self, hidden_size, attention_size=None):
        """
        Initialize the attention module with optimized dimensions.
        
        Parameters:
        -----------
        hidden_size : int
            Size of hidden states from LSTM
        attention_size : int, optional
            Size of attention layer (defaults to hidden_size)
        """
        super(AttentionModule, self).__init__()
        
        # Round to multiples of 8 for Tensor Core efficiency
        if attention_size is None:
            attention_size = ((hidden_size + 7) // 8) * 8
        else:
            attention_size = ((attention_size + 7) // 8) * 8
            
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        
        # Learnable attention parameters
        self.w_omega = nn.Parameter(torch.randn(hidden_size, attention_size))
        self.b_omega = nn.Parameter(torch.zeros(attention_size))
        self.u_omega = nn.Parameter(torch.randn(attention_size))
        
        # Initialize with scaled Xavier (Glorot) for better convergence
        nn.init.xavier_uniform_(self.w_omega, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.u_omega, gain=1/math.sqrt(2))

    def forward(self, lstm_output):
        """
        Apply scaled attention to LSTM output.
        
        Parameters:
        -----------
        lstm_output : torch.Tensor
            Output from LSTM layer [batch_size, seq_len, hidden_size]
            
        Returns:
        --------
        tuple
            (context_vector, attention_weights)
        """
        # Linear projection with scaling factor for better gradient flow
        scale = math.sqrt(self.attention_size)
        v = torch.tanh(torch.matmul(lstm_output, self.w_omega) + self.b_omega) / scale
        
        # Calculate attention scores with improved numerical stability
        vu = torch.matmul(v, self.u_omega)
        
        # Apply softmax with proper masking for numerical stability
        max_vu = torch.max(vu, dim=1, keepdim=True)[0]
        exp_vu = torch.exp(vu - max_vu)
        alphas = exp_vu / (torch.sum(exp_vu, dim=1, keepdim=True) + 1e-10)
        
        # Weighted sum of LSTM outputs based on attention
        output = torch.bmm(alphas.unsqueeze(1), lstm_output).squeeze(1)
        
        return output, alphas


class OptimizedLSTM(nn.Module):
    """
    LSTM model optimized for RTX 5090 GPU with Tensor Core support,
    mixed precision training, and optimized memory patterns.
    """
    
    def __init__(self, input_size=1, hidden_sizes=None, num_layers=3, output_size=1, 
                 dropout=0.3, bidirectional=True):
        """
        Initialize the optimized LSTM model.
        
        Parameters:
        -----------
        input_size : int
            Number of input features
        hidden_sizes : list or None
            List of hidden sizes for each layer or None for automatic sizing
        num_layers : int
            Number of LSTM layers
        output_size : int
            Size of output (prediction horizon)
        dropout : float
            Dropout rate for regularization
        bidirectional : bool
            Whether to use bidirectional LSTM
        """
        super(OptimizedLSTM, self).__init__()
        
        # Set default hidden sizes if not provided
        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]
            
        # Ensure hidden_sizes has num_layers elements
        if len(hidden_sizes) < num_layers:
            # Extend with decreasing sizes
            last_size = hidden_sizes[-1]
            while len(hidden_sizes) < num_layers:
                last_size = max(last_size // 2, 8)  # Ensure at least size 8
                hidden_sizes.append(last_size)
        
        # Truncate if too many sizes provided
        hidden_sizes = hidden_sizes[:num_layers]
        
        # Round hidden sizes to multiples of 8 for Tensor Core efficiency
        self.hidden_sizes = [((size + 7) // 8) * 8 for size in hidden_sizes]
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.bidirectional = bidirectional
        self.direction_factor = 2 if bidirectional else 1
        
        # Create LSTM layers with optimized architecture
        self.lstm_layers = nn.ModuleList()
        
        # Input shape for first layer
        layer_input_size = input_size
        
        # Create all LSTM layers
        for i, hidden_size in enumerate(self.hidden_sizes):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=layer_input_size,
                    hidden_size=hidden_size,
                    batch_first=True,
                    bidirectional=bidirectional
                )
            )
            # Update input size for next layer (accounting for bidirectional)
            layer_input_size = hidden_size * self.direction_factor
        
        # Dropout between layers (not applied after the last LSTM)
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers-1)])
        
        # Layer normalization for each LSTM output - improves training stability
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_sizes[i] * self.direction_factor) for i in range(num_layers)]
        )
        
        # Attention mechanism on the final LSTM output
        final_size = self.hidden_sizes[-1] * self.direction_factor
        self.attention = AttentionModule(final_size)
        
        # Output layers with residual connections
        self.fc1 = nn.Linear(final_size, final_size // 2)
        self.activation = nn.LeakyReLU(0.1)  # LeakyReLU for better gradient flow
        self.fc2 = nn.Linear(final_size // 2, output_size)
        
        # Dropout after first FC layer
        self.fc_dropout = nn.Dropout(dropout)
        
        # Initialize weights with optimized strategies
        self._init_weights()
        
        # Register buffers for persistent state during inference
        # This improves performance by avoiding recreating tensors
        for i, hidden_size in enumerate(self.hidden_sizes):
            # Initial hidden and cell states for each layer
            # Dimensions: (num_layers * directions, batch_size, hidden_size)
            self.register_buffer(
                f'h0_layer{i}', 
                torch.zeros(self.direction_factor, 1, hidden_size)
            )
            self.register_buffer(
                f'c0_layer{i}', 
                torch.zeros(self.direction_factor, 1, hidden_size)
            )
    
    def _init_weights(self):
        """
        Initialize weights using optimized strategies for deep LSTMs.
        Uses orthogonal initialization for recurrent weights and
        Xavier uniform for input weights.
        """
        # Initialize LSTM layers
        for layer in self.lstm_layers:
            for name, param in layer.named_parameters():
                if 'weight_ih' in name:  # Input weights
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:  # Recurrent weights
                    nn.init.orthogonal_(param.data)  # Better for RNNs
                elif 'bias' in name:  # Biases
                    # Initialize forget gate biases to 1.0 for better gradient flow
                    param.data.fill_(0.0)
                    n = param.size(0)
                    if n % 4 == 0:  # LSTM has 4 gates
                        forget_gate_idx = n // 4
                        param.data[forget_gate_idx:forget_gate_idx*2].fill_(1.0)
        
        # Initialize fully connected layers
        nn.init.xavier_uniform_(self.fc1.weight.data)
        nn.init.constant_(self.fc1.bias.data, 0.0)
        nn.init.xavier_uniform_(self.fc2.weight.data)
        nn.init.constant_(self.fc2.bias.data, 0.0)
    
    def forward(self, x):
        """
        Forward pass through the network with residual connections
        and layer normalization for improved gradient flow.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor [batch_size, seq_length, input_size]
            
        Returns:
        --------
        tuple
            (output, attention_weights)
        """
        # Get batch size for dynamic hidden state creation
        batch_size = x.size(0)
        
        # Process through LSTM layers with residual connections
        current_input = x
        hidden_states = []
        
        for i, (lstm, layer_norm) in enumerate(zip(self.lstm_layers, self.layer_norms)):
            # Expand initial states for current batch size
            h0 = getattr(self, f'h0_layer{i}').expand(-1, batch_size, -1).contiguous()
            c0 = getattr(self, f'c0_layer{i}').expand(-1, batch_size, -1).contiguous()
            
            # Pass through LSTM layer
            lstm_out, _ = lstm(current_input, (h0, c0))
            
            # Apply layer normalization for training stability
            normalized = layer_norm(lstm_out)
            
            # Store for potential residual connections
            hidden_states.append(normalized)
            
            # Apply dropout between layers (except after the last layer)
            if i < self.num_layers - 1:
                # Add residual connection if shapes match
                if i > 0 and current_input.shape == normalized.shape:
                    current_input = self.dropouts[i](normalized) + current_input
                else:
                    current_input = self.dropouts[i](normalized)
            else:
                current_input = normalized
        
        # Apply attention to the final LSTM output
        attended_output, attention_weights = self.attention(current_input)
        
        # Pass through fully connected layers with residual connection
        fc1_out = self.fc1(attended_output)
        fc1_out = self.activation(fc1_out)
        fc1_out = self.fc_dropout(fc1_out)
        
        # Final output projection
        output = self.fc2(fc1_out)
        
        return output, attention_weights


class EnsembleTrendModel(nn.Module):
    """
    Ensemble of multiple LSTM models for better prediction robustness and accuracy.
    Optimized for CPU-GPU coordination and efficient batch processing.
    """
    
    def __init__(self, input_size=1, model_configs=None, ensemble_size=3, output_size=1):
        """
        Initialize an ensemble of trend models with different configurations.
        
        Parameters:
        -----------
        input_size : int
            Number of input features
        model_configs : list of dict
            List of model configuration dictionaries
        ensemble_size : int
            Number of models in the ensemble
        output_size : int
            Size of output predictions
        """
        super(EnsembleTrendModel, self).__init__()
        
        # Default configurations with varying hyperparameters
        if model_configs is None:
            # Create diverse configurations for ensemble members
            model_configs = [
                {
                    'hidden_sizes': [128, 64, 32],
                    'num_layers': 3,
                    'dropout': 0.3,
                    'bidirectional': True
                },
                {
                    'hidden_sizes': [96, 96, 96],
                    'num_layers': 3,
                    'dropout': 0.2,
                    'bidirectional': True
                },
                {
                    'hidden_sizes': [160, 80, 40],
                    'num_layers': 3,
                    'dropout': 0.25,
                    'bidirectional': False
                }
            ]
        
        # Ensure we have enough configurations
        if len(model_configs) < ensemble_size:
            model_configs = model_configs * (ensemble_size // len(model_configs) + 1)
        model_configs = model_configs[:ensemble_size]
        
        # Create the ensemble models
        self.models = nn.ModuleList([
            OptimizedLSTM(
                input_size=input_size,
                hidden_sizes=config['hidden_sizes'],
                num_layers=config['num_layers'],
                dropout=config['dropout'],
                bidirectional=config.get('bidirectional', True),
                output_size=output_size
            ) for config in model_configs
        ])
        
        # Weight for each model (learnable)
        self.model_weights = nn.Parameter(torch.ones(ensemble_size) / ensemble_size)
        
        # Layer to combine ensemble outputs
        self.ensemble_combiner = nn.Sequential(
            nn.Linear(ensemble_size, ensemble_size),
            nn.LeakyReLU(0.1),
            nn.Linear(ensemble_size, ensemble_size)
        )
        
        # Initialize weights
        for layer in self.ensemble_combiner:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        # Ensure model is in training mode
        self.train()
    
    def forward(self, x):
        """
        Forward pass through the ensemble with dynamic weighting.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor [batch_size, seq_length, input_size]
            
        Returns:
        --------
        tuple
            (weighted_output, attention_weights_list)
        """
        batch_size = x.size(0)
        
        # Run each model
        model_outputs = []
        attention_weights = []
        
        # Process through each ensemble member
        for model in self.models:
            output, attn = model(x)
            model_outputs.append(output)
            attention_weights.append(attn)
        
        # Stack outputs
        stacked_outputs = torch.stack(model_outputs, dim=1)  # [batch_size, ensemble_size, output_size]
        
        # Apply dynamic weighting based on input
        # This allows the ensemble to adapt weights based on the specific input pattern
        normalized_weights = torch.softmax(self.model_weights, dim=0)
        
        # Process through ensemble combiner if batch size > 1
        if batch_size > 1:
            # Get dynamic weights for this specific input
            dynamic_weights = torch.softmax(self.ensemble_combiner(normalized_weights), dim=0)
            # Apply dynamic weights
            weighted_output = torch.sum(stacked_outputs * dynamic_weights.view(1, -1, 1), dim=1)
        else:
            # Simple weighted average for single samples
            weighted_output = torch.sum(stacked_outputs * normalized_weights.view(1, -1, 1), dim=1)
        
        return weighted_output, attention_weights


class TrendPredictionModule:
    """
    A comprehensive module for trend prediction using optimized LSTMs with CUDA enhancements
    specifically tuned for RTX 5090 GPUs. Includes mixed precision training, gradient 
    accumulation, and memory-optimized batch processing.
    """
    
    def __init__(self, model, learning_rate=0.001, weight_decay=1e-5, device=None, use_custom_loss=True):
        """
        Initialize the trend prediction module.
        
        Parameters:
        -----------
        model : nn.Module
            The LSTM-based model for trend prediction
        learning_rate : float
            Learning rate for optimization
        weight_decay : float
            L2 regularization coefficient
        device : torch.device or None
            Device to use (auto-detects if None)
        """
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Log device information
        if self.device.type == 'cuda':
            logger.info(f"Using GPU: {torch.cuda.get_device_name(self.device)}")
            # Enable TF32 precision for better performance on Ampere and later GPUs
            if torch.cuda.get_device_capability(self.device)[0] >= 8:
                logger.info("Enabling TF32 precision for matrix multiplications")
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        else:
            logger.info("Using CPU for computations")
            
        # Move model to device
        self.model = model.to(self.device)
        
        # Setup optimizer with learning rate and weight decay
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Setup loss function - either custom or standard MSE
        if use_custom_loss:
            self.loss_fn = custom_return_maximizing_loss
            logger.info("Using custom return-maximizing loss function")
        else:
            self.loss_fn = nn.MSELoss()
            logger.info("Using standard MSE loss function")
        
        # Setup gradient scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == 'cuda')
        
        # Initialize training stats
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.steps = 0
        
        # Benchmark for auto-tuning
        if self.device.type == 'cuda':
            logger.info("Enabling cuDNN benchmark for auto-tuning")
            torch.backends.cudnn.benchmark = True
        
       # Skip compilation for LSTM models - they're not supported by TorchDynamo
        if isinstance(self.model, nn.LSTM) or hasattr(self.model, 'lstm'):
            logger.info("Skipping compilation for LSTM model - not supported by TorchDynamo")
            self.compiled_model = self.model
            self.use_compiled = False
        # Try to use torch.compile() if available for other model types (PyTorch 2.0+)
        elif hasattr(torch, 'compile') and self.device.type == 'cuda':
            try:
                logger.info("Compiling model with torch.compile()...")
                self.compiled_model = torch.compile(
                    self.model, 
                    mode="max-autotune",
                    fullgraph=True
                )
                self.use_compiled = True
                logger.info("Model successfully compiled")
            except Exception as e:
                logger.warning(f"Model compilation failed: {str(e)}")
                self.compiled_model = self.model
                self.use_compiled = False
        else:
            self.compiled_model = self.model
            self.use_compiled = False
            
    def train_batch(self, inputs, targets, accumulation_steps=1):
        """
        Train on a single batch with mixed precision and gradient accumulation.
        
        Parameters:
        -----------
        inputs : torch.Tensor
            Input tensor
        targets : torch.Tensor
            Target tensor
        accumulation_steps : int
            Number of gradient accumulation steps
            
        Returns:
        --------
        float
            Loss value
        """
        # Move data to device
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        
        # Ensure targets are shaped correctly
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        
        # Use the active model (compiled or original)
        active_model = self.compiled_model if self.use_compiled else self.model
        
        # Set model to training mode
        active_model.train()
        
        # Forward pass with mixed precision
        with torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
            outputs, _ = active_model(inputs)
            loss = self.loss_fn(outputs, targets) / accumulation_steps
        
        # Backward pass with scaling
        self.scaler.scale(loss).backward()
        
        # Only update weights after accumulation_steps
        self.steps += 1
        if self.steps % accumulation_steps == 0:
            # Unscale gradients for proper weight decay
            self.scaler.unscale_(self.optimizer)
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(active_model.parameters(), max_norm=1.0)
            
            # Step optimizer and update scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Zero gradients (more efficient than optimizer.zero_grad())
            for param in active_model.parameters():
                param.grad = None
        
        # Return full loss value (not divided by accumulation_steps)
        return loss.item() * accumulation_steps
    
    def evaluate(self, inputs, targets):
        """
        Evaluate the model on validation data.
        
        Parameters:
        -----------
        inputs : torch.Tensor
            Input tensor
        targets : torch.Tensor
            Target tensor
            
        Returns:
        --------
        tuple
            (loss, predictions, attention_weights)
        """
        # Move data to device
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        
        # Ensure targets are shaped correctly
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        
        # Use the active model (compiled or original)
        active_model = self.compiled_model if self.use_compiled else self.model
        
        # Set model to evaluation mode
        active_model.eval()
        
        # Evaluate with no gradients
        with torch.no_grad():
            # Use mixed precision for evaluation too
            with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
                predictions, attention_weights = active_model(inputs)
                loss = self.loss_fn(predictions, targets).item()
        
        return loss, predictions.cpu().numpy(), attention_weights
    
    def predict(self, inputs, use_fp16=False):
        """
        Make predictions with optimized inference.
        
        Parameters:
        -----------
        inputs : torch.Tensor
            Input tensor
        use_fp16 : bool
            Whether to use FP16 precision for faster inference
            
        Returns:
        --------
        tuple
            (predictions, attention_weights)
        """
        # Move inputs to device
        inputs = inputs.to(self.device, non_blocking=True)
        
        # Use the active model (compiled or original)
        active_model = self.compiled_model if self.use_compiled else self.model
        
        # Set model to evaluation mode
        active_model.eval()
        
        # Make predictions with no gradients
        with torch.no_grad():
            if use_fp16 and self.device.type == 'cuda':
                # Use FP16 precision for faster inference
                with torch.cuda.amp.autocast():
                    predictions, attention_weights = active_model(inputs)
            else:
                # Use FP32 precision
                predictions, attention_weights = active_model(inputs)
        
        # Return predictions and attention weights
        return predictions.cpu().numpy(), attention_weights
    
    # In TrendPredictionModule.train_epochs method

    def train_epochs(self, train_loader, val_loader, epochs=50, accumulation_steps=4, patience=10):
        """
        Train the model for multiple epochs with early stopping.
        
        Parameters:
        -----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader
            Validation data loader
        epochs : int
            Number of training epochs
        accumulation_steps : int
            Number of gradient accumulation steps
        patience : int
            Early stopping patience
                
        Returns:
        --------
        tuple
            (train_losses, val_losses, best_val_loss)
        """
        # Initialize early stopping variables
        patience_counter = 0
        best_val_loss = float('inf')
        best_weights = None
        
        train_losses = []
        val_losses = []
        
        # Log start of training
        logger.info(f"Starting training for {epochs} epochs (patience={patience})")
        
        # Training loop with early stopping
        for epoch in range(epochs):
            # Training phase
            train_loss = 0.0
            num_batches = 0
            
            # Log epoch start
            logger.info(f"Epoch {epoch+1}/{epochs}: Training...")
            epoch_start_time = time.time()
            
            # Train on batches
            for inputs, targets in train_loader:
                batch_loss = self.train_batch(inputs, targets, accumulation_steps)
                train_loss += batch_loss
                num_batches += 1
            
            # Calculate average training loss
            train_loss /= max(num_batches, 1)
            train_losses.append(train_loss)
            
            # Validation phase
            val_loss = 0.0
            num_val_batches = 0
            
            # Log validation start
            logger.info(f"Epoch {epoch+1}/{epochs}: Validating...")
            
            # Evaluate on validation batches
            for inputs, targets in val_loader:
                batch_loss, _, _ = self.evaluate(inputs, targets)
                val_loss += batch_loss
                num_val_batches += 1
            
            # Calculate average validation loss
            val_loss /= max(num_val_batches, 1)
            val_losses.append(val_loss)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Print progress
            logger.info(f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.1f}s: "
                    f"Train Loss = {train_loss:.6f}, "
                    f"Val Loss = {val_loss:.6f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save best model weights
                best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
                logger.info(f"New best validation loss: {val_loss:.6f}")
            else:
                patience_counter += 1
                logger.info(f"No improvement for {patience_counter}/{patience} epochs")
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best weights
        if best_weights is not None:
            self.model.load_state_dict(best_weights)
            logger.info("Loaded best model weights")
        
        # Set class attributes
        self.train_losses = train_losses
        self.val_losses = val_losses
        self.best_val_loss = best_val_loss
        
        # Return training statistics
        return train_losses, val_losses, best_val_loss
    
    def optimize_batch_size(self, seq_length, feature_dim, target_memory=0.7):
        """
        Find optimal batch size based on GPU memory.
        
        Parameters:
        -----------
        seq_length : int
            Sequence length of input data
        feature_dim : int
            Feature dimension of input data
        target_memory : float
            Target memory utilization (0.0-1.0)
            
        Returns:
        --------
        int
            Optimal batch size
        """
        if self.device.type != 'cuda':
            # Default for CPU
            return 32
        
        # Estimate available GPU memory
        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        free_memory = total_gpu_memory * target_memory
        
        # Estimate model memory usage
        model_size = sum(p.numel() * p.element_size() 
                        for p in self.model.parameters()) / 1e9
        
        # Estimate input size per sample
        sample_size = seq_length * feature_dim * 4 / 1e9  # FP32 size
        
        # Factor in gradient and optimizer state memory
        # Typically 4x the model size for optimizer states, gradients
        available_memory = free_memory - model_size * 5
        
        # Calculate optimal batch size (with a safety factor of 0.8)
        optimal_batch = max(1, int(available_memory / (sample_size * 4) * 0.8))
        
        logger.info(f"Estimated optimal batch size: {optimal_batch} "
                  f"(GPU: {total_gpu_memory:.1f}GB, Model: {model_size:.2f}GB)")
        
        # Round to power of 2 for better GPU utilization
        power_of_2 = 2 ** int(np.log2(optimal_batch))
        return power_of_2
    
    def create_cuda_graph(self, sample_input):
        """
        Create CUDA graph for optimal inference performance.
        
        Parameters:
        -----------
        sample_input : torch.Tensor
            Sample input tensor for graph capture
            
        Returns:
        --------
        tuple
            (cuda_graph, static_input, static_output)
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping graph creation")
            return None, None, None
        
        try:
            # Ensure model is in eval mode
            self.model.eval()
            
            # Create static sample on device
            static_input = sample_input.to(self.device).clone()
            
            # Warmup
            self.model(static_input)
            
            # Capture CUDA graph
            cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cuda_graph):
                static_output, static_attention = self.model(static_input)
            
            logger.info("CUDA graph captured successfully")
            return cuda_graph, static_input, (static_output, static_attention)
            
        except Exception as e:
            logger.error(f"CUDA graph capture failed: {str(e)}")
            return None, None, None
    
    def inference_with_graph(self, inputs, cuda_graph, static_input, static_outputs):
        """
        Perform inference using pre-captured CUDA graph for maximum performance.
        
        Parameters:
        -----------
        inputs : torch.Tensor
            Input tensor
        cuda_graph : torch.cuda.CUDAGraph
            Pre-captured CUDA graph
        static_input : torch.Tensor
            Static input tensor used for graph capture
        static_outputs : tuple
            Static output tensors from graph capture
            
        Returns:
        --------
        tuple
            (predictions, attention_weights)
        """
        if cuda_graph is None:
            # Fall back to regular inference
            return self.predict(inputs)
        
        # Ensure inputs match the expected shape
        if inputs.shape != static_input.shape:
            logger.warning(f"Input shape mismatch: got {inputs.shape}, "
                          f"expected {static_input.shape}")
            # Fall back to regular inference
            return self.predict(inputs)
        
        # Copy inputs to static tensor
        static_input.copy_(inputs.to(self.device))
        
        # Run the graph
        cuda_graph.replay()
        
        # Get results (clone to avoid modifying the static tensors)
        static_output, static_attention = static_outputs
        predictions = static_output.clone().cpu().numpy()
        attention_weights = static_attention
        
        return predictions, attention_weights
    
    def benchmark_inference(self, sample_input, iterations=100):
        """
        Benchmark inference performance with different optimization techniques.
        
        Parameters:
        -----------
        sample_input : torch.Tensor
            Sample input tensor
        iterations : int
            Number of inference iterations
            
        Returns:
        --------
        dict
            Benchmark results
        """
        import time
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            logger.info("CUDA not available, skipping benchmark")
            return {"cpu_time": self._run_benchmark(sample_input, iterations, use_gpu=False)}
        
        # Ensure model is in eval mode
        self.model.eval()
        
        # Run benchmarks with different configurations
        results = {}
        
        # 1. CPU inference
        results["cpu_time"] = self._run_benchmark(
            sample_input, iterations, use_gpu=False
        )
        
        # 2. GPU inference with FP32
        results["gpu_fp32_time"] = self._run_benchmark(
            sample_input, iterations, use_gpu=True, use_fp16=False, use_graph=False
        )
        
        # 3. GPU inference with FP16
        results["gpu_fp16_time"] = self._run_benchmark(
            sample_input, iterations, use_gpu=True, use_fp16=True, use_graph=False
        )
        
        # 4. GPU inference with CUDA graph (if available)
        if torch.cuda.get_device_capability(0)[0] >= 7:  # Volta or later
            # Create CUDA graph
            cuda_graph, static_input, static_outputs = self.create_cuda_graph(sample_input)
            
            if cuda_graph is not None:
                # Run benchmark with CUDA graph
                results["gpu_graph_time"] = self._run_benchmark(
                    sample_input, iterations, use_gpu=True, use_fp16=False, 
                    use_graph=True, cuda_graph=cuda_graph, 
                    static_input=static_input, static_outputs=static_outputs
                )
        
        # 5. Compiled model inference (if available)
        if self.use_compiled:
            results["compiled_model_time"] = self._run_benchmark(
                sample_input, iterations, use_gpu=True, use_fp16=True,
                use_graph=False, use_compiled=True
            )
        
        # Calculate speedups
        cpu_time = results["cpu_time"]
        for key, time_value in results.items():
            if key != "cpu_time":
                results[f"{key}_speedup"] = cpu_time / time_value
        
        return results
    
    def _run_benchmark(self, sample_input, iterations, use_gpu=True, use_fp16=False, 
                      use_graph=False, cuda_graph=None, static_input=None, 
                      static_outputs=None, use_compiled=False):
        """
        Run inference benchmark with specific configuration.
        
        Parameters:
        -----------
        sample_input : torch.Tensor
            Sample input tensor
        iterations : int
            Number of inference iterations
        use_gpu : bool
            Whether to use GPU
        use_fp16 : bool
            Whether to use FP16 precision
        use_graph : bool
            Whether to use CUDA graph
        cuda_graph : torch.cuda.CUDAGraph or None
            Pre-captured CUDA graph
        static_input : torch.Tensor or None
            Static input tensor for graph capture
        static_outputs : tuple or None
            Static output tensors from graph capture
        use_compiled : bool
            Whether to use compiled model
            
        Returns:
        --------
        float
            Average inference time per iteration (ms)
        """
        import time
        
        # Determine device
        device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        
        # Select model
        if use_compiled and self.use_compiled:
            active_model = self.compiled_model
        else:
            active_model = self.model
        
        # Prepare input tensor
        input_tensor = sample_input.to(device if not use_graph else 'cpu')
        
        # Warmup
        for _ in range(10):
            if use_graph:
                # Use CUDA graph for inference
                static_input.copy_(input_tensor.to(device))
                cuda_graph.replay()
            else:
                # Regular inference
                with torch.no_grad():
                    if use_fp16 and device.type == 'cuda':
                        with torch.cuda.amp.autocast():
                            _ = active_model(input_tensor)
                    else:
                        _ = active_model(input_tensor)
        
        # Synchronize before timing
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Start timing
        start_time = time.time()
        
        # Run inference iterations
        for _ in range(iterations):
            if use_graph:
                # Use CUDA graph for inference
                static_input.copy_(input_tensor.to(device))
                cuda_graph.replay()
            else:
                # Regular inference
                with torch.no_grad():
                    if use_fp16 and device.type == 'cuda':
                        with torch.cuda.amp.autocast():
                            _ = active_model(input_tensor)
                    else:
                        _ = active_model(input_tensor)
        
        # Synchronize after timing
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        avg_time_ms = (elapsed_time / iterations) * 1000
        
        config_str = f"{'GPU' if device.type == 'cuda' else 'CPU'}"
        config_str += f"{' + FP16' if use_fp16 else ''}"
        config_str += f"{' + Graph' if use_graph else ''}"
        config_str += f"{' + Compiled' if use_compiled and self.use_compiled else ''}"
        
        logger.info(f"Benchmark [{config_str}]: {avg_time_ms:.3f} ms per inference")
        
        return avg_time_ms