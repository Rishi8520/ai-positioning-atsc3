"""
Improved Broadcast Decision Model - Version 2.0
Features:
- Residual connections for deeper learning
- Monte Carlo Dropout for uncertainty quantification
- Multi-task loss for balanced optimization
- Improved training metrics
"""

import logging
import json
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ==============================================================================
# LOGGING
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTS
# ==============================================================================
INPUT_DIM = 50
HIDDEN_DIMS = [128, 64, 32]
OUTPUT_DIM = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 200
VALIDATION_SPLIT = 0.15
EARLY_STOPPING_PATIENCE = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Output normalization ranges
OUTPUT_RANGES = {
    0: (1.0, 5.0),      # redundancy_ratio
    1: (0.1, 2.0),      # spectrum_mbps
    2: (0.80, 0.99),    # availability_pct
    3: (10.0, 60.0),    # convergence_time_sec
    4: (1.0, 50.0)      # accuracy_hpe_cm
}

# Multi-task loss weights
LOSS_WEIGHTS = {
    'redundancy': 1.0,
    'spectrum': 1.5,      # Spectrum is critical
    'availability': 2.0,  # Availability is most important
    'convergence': 1.0,
    'accuracy': 1.5       # Accuracy is critical
}

# ==============================================================================
# DATA CLASSES
# ==============================================================================
@dataclass
class BroadcastDecision:
    """Output from decision model"""
    redundancy_ratio: float
    spectrum_mbps: float
    availability_pct: float
    convergence_time_sec: float
    accuracy_hpe_cm: float
    confidence: float
    uncertainty: float = 0.0  # NEW: Uncertainty estimate
    
    def to_dict(self):
        return {
            "redundancy_ratio": self.redundancy_ratio,
            "spectrum_mbps": self.spectrum_mbps,
            "availability_pct": self.availability_pct,
            "convergence_time_sec": self.convergence_time_sec,
            "accuracy_hpe_cm": self.accuracy_hpe_cm,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty
        }

# ==============================================================================
# IMPROVED NEURAL NETWORK WITH RESIDUAL CONNECTIONS
# ==============================================================================
class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    def __init__(self, dim: int, dropout_rate: float = 0.3):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += residual  # Skip connection
        out = self.relu(out)
        return out

class BroadcastDecisionNetV2(nn.Module):
    """
    Improved neural network with residual connections and MC Dropout
    
    Architecture:
    - Input: 50D → 128D → 128D (residual) → 64D → 32D → 5D
    - Residual blocks for better gradient flow
    - MC Dropout for uncertainty estimation
    """
    def __init__(self, input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS, 
                 output_dim=OUTPUT_DIM, dropout_rate=0.3):
        super(BroadcastDecisionNetV2, self).__init__()
        
        # Input projection
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Residual blocks at first hidden layer
        self.residual_block1 = ResidualBlock(hidden_dims[0], dropout_rate)
        self.residual_block2 = ResidualBlock(hidden_dims[0], dropout_rate)
        
        # Hidden layers with skip connections
        self.hidden1 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.hidden2 = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Output layer (separate heads for each output)
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_dims[2], 1) for _ in range(output_dim)
        ])
        
        self.sigmoid = nn.Sigmoid()
        
        logger.info(f"BroadcastDecisionNetV2 initialized: {input_dim} -> {hidden_dims} -> {output_dim}")
        logger.info(f"Using residual connections and MC Dropout")
    
    def forward(self, x, mc_samples=1):
        """
        Forward pass with optional MC Dropout sampling
        
        Args:
            x: Input tensor (batch_size, input_dim)
            mc_samples: Number of MC Dropout samples for uncertainty
        
        Returns:
            If mc_samples=1: (batch_size, output_dim)
            If mc_samples>1: (mc_samples, batch_size, output_dim)
        """
        if mc_samples == 1:
            # Standard forward pass
            out = self.input_layer(x)
            out = self.residual_block1(out)
            out = self.residual_block2(out)
            out = self.hidden1(out)
            out = self.hidden2(out)
            
            # Multi-head output
            outputs = [head(out) for head in self.output_heads]
            out = torch.cat(outputs, dim=1)
            out = self.sigmoid(out)
            return out
        else:
            # MC Dropout: multiple forward passes with dropout enabled
            # Keep BatchNorm in eval mode, but enable dropout manually
            self.eval()  # Keep batch norm in eval mode
            
            # Manually enable dropout layers
            def enable_dropout(m):
                if type(m) == nn.Dropout:
                    m.train()
            
            self.apply(enable_dropout)
            
            predictions = []
            with torch.no_grad():
                for _ in range(mc_samples):
                    out = self.input_layer(x)
                    out = self.residual_block1(out)
                    out = self.residual_block2(out)
                    out = self.hidden1(out)
                    out = self.hidden2(out)
                    
                    outputs = [head(out) for head in self.output_heads]
                    out = torch.cat(outputs, dim=1)
                    out = self.sigmoid(out)
                    predictions.append(out)
            
            return torch.stack(predictions)

# ==============================================================================
# MULTI-TASK LOSS FUNCTION
# ==============================================================================
class MultiTaskLoss(nn.Module):
    """Weighted multi-task loss for balanced optimization"""
    def __init__(self, weights=LOSS_WEIGHTS):
        super(MultiTaskLoss, self).__init__()
        self.weights = weights
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(self, predictions, targets):
        """
        Compute weighted MSE loss per output dimension
        
        Args:
            predictions: (batch_size, 5)
            targets: (batch_size, 5)
        
        Returns:
            Scalar loss
        """
        # Compute MSE per output dimension
        losses = self.mse(predictions, targets)  # (batch_size, 5)
        
        # Apply weights
        weights_tensor = torch.tensor([
            self.weights['redundancy'],
            self.weights['spectrum'],
            self.weights['availability'],
            self.weights['convergence'],
            self.weights['accuracy']
        ], device=predictions.device)
        
        weighted_losses = losses * weights_tensor
        
        # Average over batch and dimensions
        total_loss = weighted_losses.mean()
        
        return total_loss, losses.mean(dim=0)  # Return total and per-dimension

# ==============================================================================
# MODEL TRAINER WITH IMPROVED METRICS
# ==============================================================================
class ModelTrainerV2:
    """Improved trainer with better metrics and validation"""
    
    def __init__(self, model=None, learning_rate=LEARNING_RATE, device=DEVICE):
        self.device = torch.device(device)
        
        if model is None:
            self.model = BroadcastDecisionNetV2().to(self.device)
        else:
            self.model = model.to(self.device)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=NUM_EPOCHS)
        self.criterion = MultiTaskLoss()
        
        # Scalers for normalization
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        self.scaler_X = StandardScaler()   # StandardScaler for inputs
        self.scaler_y = MinMaxScaler()     # MinMaxScaler for outputs (compatible with sigmoid)
        
        logger.info(f"ModelTrainerV2 initialized on device: {self.device}")
        logger.info(f"Optimizer: AdamW with cosine annealing scheduler")
        logger.info(f"Using StandardScaler for X, MinMaxScaler for y")
    
    def prepare_data(self, X, y, batch_size=BATCH_SIZE):
        """
        Prepare data for training
        
        Args:
            X: (N, 50) input features
            y: (N, 5) target outputs
            batch_size: Batch size
        
        Returns:
            train_loader, scaler_X, scaler_y
        """
        # Normalize
        X_normalized = self.scaler_X.fit_transform(X)
        y_normalized = self.scaler_y.fit_transform(y)
        
        # Convert to tensors
        X_tensor = torch.from_numpy(X_normalized).float()
        y_tensor = torch.from_numpy(y_normalized).float()
        
        # Create dataset and loader
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        logger.info(f"Data prepared: {len(dataset)} samples, batch_size={batch_size}")
        return loader
    
    def train(self, train_loader, val_loader, num_epochs=NUM_EPOCHS, 
              patience=EARLY_STOPPING_PATIENCE):
        """
        Train the model with early stopping
        
        Returns:
            Dictionary with training history
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_mae": [],
            "val_mae": [],
            "lr": []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_mae = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss, per_dim_loss = self.criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                train_loss += loss.item()
                train_mae += torch.abs(outputs - batch_y).mean().item()
            
            train_loss /= len(train_loader)
            train_mae /= len(train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_mae = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_X)
                    loss, _ = self.criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    val_mae += torch.abs(outputs - batch_y).mean().item()
            
            val_loss /= len(val_loader)
            val_mae /= len(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_mae"].append(train_mae)
            history["val_mae"].append(val_mae)
            history["lr"].append(current_lr)
            
            # Logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Train Loss: {train_loss:.6f}, MAE: {train_mae:.6f} | "
                    f"Val Loss: {val_loss:.6f}, MAE: {val_mae:.6f} | "
                    f"LR: {current_lr:.6f}"
                )
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                logger.info(f"✓ New best model at epoch {epoch+1}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info("Restored best model weights")
        
        logger.info("Training complete!")
        return history
    
    def save_model(self, path: str):
        """Save model and scalers"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.model.state_dict(), path / "model_weights.pt")
        pickle.dump(self.scaler_X, open(path / "scaler_X.pkl", "wb"))
        pickle.dump(self.scaler_y, open(path / "scaler_y.pkl", "wb"))
        
        # Save model config
        config = {
            'input_dim': INPUT_DIM,
            'hidden_dims': HIDDEN_DIMS,
            'output_dim': OUTPUT_DIM,
            'output_ranges': OUTPUT_RANGES,
            'device': str(self.device)
        }
        with open(path / "model_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model and scalers"""
        path = Path(path)
        
        self.model.load_state_dict(torch.load(path / "model_weights.pt", map_location=self.device))
        self.scaler_X = pickle.load(open(path / "scaler_X.pkl", "rb"))
        self.scaler_y = pickle.load(open(path / "scaler_y.pkl", "rb"))
        
        logger.info(f"Model loaded from {path}")

# ==============================================================================
# INFERENCE ENGINE WITH UNCERTAINTY
# ==============================================================================
class DecisionInferenceEngineV2:
    """Inference engine with uncertainty quantification"""
    
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BroadcastDecisionNetV2().to(self.device)
        
        # Load model and scalers
        path = Path(model_path)
        self.model.load_state_dict(torch.load(path / "model_weights.pt", map_location=self.device))
        self.scaler_X = pickle.load(open(path / "scaler_X.pkl", "rb"))
        self.scaler_y = pickle.load(open(path / "scaler_y.pkl", "rb"))
        
        self.model.eval()
        logger.info(f"DecisionInferenceEngineV2 initialized from {model_path}")
    
    def infer(self, telemetry_features: np.ndarray, mc_samples=20) -> BroadcastDecision:
        """
        Make decision with uncertainty quantification
        
        Args:
            telemetry_features: (50,) array
            mc_samples: Number of MC Dropout samples
        
        Returns:
            BroadcastDecision with uncertainty
        """
        if telemetry_features.shape != (INPUT_DIM,):
            raise ValueError(f"Expected shape ({INPUT_DIM},), got {telemetry_features.shape}")
        
        # Normalize
        X_normalized = self.scaler_X.transform(telemetry_features.reshape(1, -1))
        X_tensor = torch.from_numpy(X_normalized).float().to(self.device)
        
        # MC Dropout inference
        predictions = self.model(X_tensor, mc_samples=mc_samples)  # (mc_samples, 1, 5)
        predictions = predictions.squeeze(1).cpu().numpy()  # (mc_samples, 5)
        
        # Compute mean and std
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        # Denormalize
        mean_denorm = self.scaler_y.inverse_transform(mean_pred.reshape(1, -1))[0]
        
        # Clip to valid ranges
        decision_values = []
        for idx, (min_val, max_val) in OUTPUT_RANGES.items():
            clipped = np.clip(mean_denorm[idx], min_val, max_val)
            decision_values.append(clipped)
        
        # Confidence: inverse of normalized uncertainty
        uncertainty = std_pred.mean()
        confidence = 1.0 / (1.0 + uncertainty)
        confidence = np.clip(confidence, 0.0, 1.0)
        
        decision = BroadcastDecision(
            redundancy_ratio=decision_values[0],
            spectrum_mbps=decision_values[1],
            availability_pct=decision_values[2],
            convergence_time_sec=decision_values[3],
            accuracy_hpe_cm=decision_values[4],
            confidence=float(confidence),
            uncertainty=float(uncertainty)
        )
        
        return decision

# ==============================================================================
# MAIN TRAINING SCRIPT
# ==============================================================================
if __name__ == "__main__":
    logger.info("="*80)
    logger.info("BROADCAST DECISION MODEL V2 - TRAINING")
    logger.info("="*80)
    
    # Load generated dataset
    data_dir = Path(__file__).parent.parent.parent / "DATA"
    
    logger.info(f"Loading dataset from {data_dir}...")
    X_train = np.load(data_dir / 'X_train.npy')
    y_train = np.load(data_dir / 'y_train.npy')
    X_val = np.load(data_dir / 'X_val.npy')
    y_val = np.load(data_dir / 'y_val.npy')
    
    logger.info(f"Train: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Val:   X={X_val.shape}, y={y_val.shape}")
    
    # Create trainer
    trainer = ModelTrainerV2()
    
    # Prepare data
    train_loader = trainer.prepare_data(X_train, y_train)
    val_loader = trainer.prepare_data(X_val, y_val)
    
    # Train model
    history = trainer.train(train_loader, val_loader, num_epochs=NUM_EPOCHS)
    
    # Save model
    model_path = Path(__file__).parent / "models" / "broadcast_decision_model_v2"
    trainer.save_model(str(model_path))
    
    # Save training history
    with open(model_path / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Final val loss: {history['val_loss'][-1]:.6f}")
    logger.info(f"Final val MAE: {history['val_mae'][-1]:.6f}")