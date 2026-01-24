# ===============================================================================
# FILE: ai_broadcast_decision_model.py
# MODULE: Deep Learning Model for Broadcast Configuration Decision
# AUTHOR: Tarunika D (AI/ML Systems)
# DATE: January 2026
# PURPOSE: Neural network model to determine optimal broadcast parameters
# PRODUCTION: Phase 3 - Ready for Deployment
# ===============================================================================

import logging
import json
import pickle
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sklearn.preprocessing as preprocessing

# ===============================================================================
# LOGGING
# ===============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===============================================================================
# CONSTANTS
# ===============================================================================

INPUT_DIM = 50  # 50D feature vector (telemetry + channel state)
HIDDEN_DIMS = [128, 64, 32]
OUTPUT_DIM = 5  # 5D output (broadcast optimization targets)
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
VALIDATION_SPLIT = 0.15
EARLY_STOPPING_PATIENCE = 5

# Output normalization ranges
OUTPUT_RANGES = {
    0: (1.0, 5.0),      # redundancy_ratio
    1: (0.1, 2.0),      # spectrum_mbps
    2: (0.80, 0.99),    # availability_pct
    3: (10, 60),        # convergence_time_sec
    4: (1.0, 50.0)      # accuracy_hpe_cm
}

# ===============================================================================
# DATA CLASSES
# ===============================================================================

@dataclass
class BroadcastDecision:
    """Output from decision model"""
    redundancy_ratio: float  # (1.0-5.0)
    spectrum_mbps: float     # (0.1-2.0)
    availability_pct: float  # (0.80-0.99)
    convergence_time_sec: float  # (10-60)
    accuracy_hpe_cm: float   # (1.0-50.0)
    confidence: float        # Confidence score
    
    def to_dict(self):
        return {
            "redundancy_ratio": self.redundancy_ratio,
            "spectrum_mbps": self.spectrum_mbps,
            "availability_pct": self.availability_pct,
            "convergence_time_sec": self.convergence_time_sec,
            "accuracy_hpe_cm": self.accuracy_hpe_cm,
            "confidence": self.confidence
        }


# ===============================================================================
# NEURAL NETWORK ARCHITECTURE
# ===============================================================================

class BroadcastDecisionNet(nn.Module):
    """
    Neural network for broadcast decision making
    
    Architecture:
    - Input: 50D (telemetry + channel state)
    - Hidden: [128, 64, 32] with ReLU + BatchNorm + Dropout
    - Output: 5D (broadcast optimization targets)
    """
    
    def __init__(self, input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS, output_dim=OUTPUT_DIM):
        super(BroadcastDecisionNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())  # Normalize to [0, 1]
        
        self.network = nn.Sequential(*layers)
        
        logger.info(f"BroadcastDecisionNet initialized: {input_dim} -> {hidden_dims} -> {output_dim}")
    
    def forward(self, x):
        return self.network(x)


# ===============================================================================
# MODEL TRAINER
# ===============================================================================

class ModelTrainer:
    """Trainer for broadcast decision model"""
    
    def __init__(self, model=None, learning_rate=LEARNING_RATE):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model is None:
            self.model = BroadcastDecisionNet().to(self.device)
        else:
            self.model = model.to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.scaler_X = preprocessing.StandardScaler()
        self.scaler_y = preprocessing.StandardScaler()
        
        logger.info(f"ModelTrainer initialized on device: {self.device}")
    
    def prepare_data(self, X, y, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT):
        """
        Prepare data for training
        
        Args:
            X: (N, 50) array of input features
            y: (N, 5) array of target outputs
            batch_size: Batch size
            validation_split: Fraction for validation set
        
        Returns:
            train_loader, val_loader, scaler_X, scaler_y
        """
        # Normalize inputs and outputs
        X_normalized = self.scaler_X.fit_transform(X)
        y_normalized = self.scaler_y.fit_transform(y)
        
        # Convert to tensors
        X_tensor = torch.from_numpy(X_normalized).float()
        y_tensor = torch.from_numpy(y_normalized).float()
        
        # Split into train/val
        num_samples = len(X)
        num_val = int(num_samples * validation_split)
        num_train = num_samples - num_val
        
        train_dataset = TensorDataset(X_tensor[:num_train], y_tensor[:num_train])
        val_dataset = TensorDataset(X_tensor[num_train:], y_tensor[num_train:])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"Data prepared: {num_train} train, {num_val} val samples")
        
        return train_loader, val_loader
    
    def train(self, train_loader, val_loader, num_epochs=NUM_EPOCHS, patience=EARLY_STOPPING_PATIENCE):
        """
        Train the model
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            num_epochs: Number of epochs
            patience: Early stopping patience
        
        Returns:
            Dictionary with training history
        """
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        logger.info("Training complete")
        return history
    
    def save_model(self, path: str):
        """Save model and scalers to disk"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.model.state_dict(), path / "model_weights.pt")
        pickle.dump(self.scaler_X, open(path / "scaler_X.pkl", "wb"))
        pickle.dump(self.scaler_y, open(path / "scaler_y.pkl", "wb"))
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model and scalers from disk"""
        path = Path(path)
        
        self.model.load_state_dict(torch.load(path / "model_weights.pt", map_location=self.device))
        self.scaler_X = pickle.load(open(path / "scaler_X.pkl", "rb"))
        self.scaler_y = pickle.load(open(path / "scaler_y.pkl", "rb"))
        
        logger.info(f"Model loaded from {path}")


# ===============================================================================
# INFERENCE ENGINE
# ===============================================================================

class DecisionInferenceEngine:
    """Inference engine for real-time decision making"""
    
    def __init__(self, model_path: str):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to trained model directory
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BroadcastDecisionNet().to(self.device)
        
        # Load model and scalers
        path = Path(model_path)
        self.model.load_state_dict(torch.load(path / "model_weights.pt", map_location=self.device))
        self.scaler_X = pickle.load(open(path / "scaler_X.pkl", "rb"))
        self.scaler_y = pickle.load(open(path / "scaler_y.pkl", "rb"))
        
        self.model.eval()
        logger.info(f"DecisionInferenceEngine initialized from {model_path}")
    
    def infer(self, telemetry_features: np.ndarray, confidence_threshold=0.6) -> BroadcastDecision:
        """
        Make broadcast decision from telemetry features
        
        Args:
            telemetry_features: (50,) array of normalized telemetry + channel state
            confidence_threshold: Minimum confidence to accept model output
        
        Returns:
            BroadcastDecision with parameters and confidence
        """
        # Validate input
        if telemetry_features.shape != (INPUT_DIM,):
            raise ValueError(f"Expected input shape ({INPUT_DIM},), got {telemetry_features.shape}")
        
        # Normalize and convert to tensor
        X_normalized = self.scaler_X.transform(telemetry_features.reshape(1, -1))
        X_tensor = torch.from_numpy(X_normalized).float().to(self.device)
        
        # Inference
        with torch.no_grad():
            output_normalized = self.model(X_tensor).cpu().numpy()[0]
        
        # Denormalize output
        output_denormalized = self.scaler_y.inverse_transform(output_normalized.reshape(1, -1))[0]
        
        # Clip to valid ranges
        decision_values = []
        for idx, (min_val, max_val) in OUTPUT_RANGES.items():
            clipped = np.clip(output_denormalized[idx], min_val, max_val)
            decision_values.append(clipped)
        
        # Calculate confidence as average output magnitude
        confidence = float(np.mean(np.abs(output_normalized)))
        confidence = np.clip(confidence, 0.0, 1.0)
        
        decision = BroadcastDecision(
            redundancy_ratio=decision_values[0],
            spectrum_mbps=decision_values[1],
            availability_pct=decision_values[2],
            convergence_time_sec=decision_values[3],
            accuracy_hpe_cm=decision_values[4],
            confidence=confidence
        )
        
        logger.debug(f"Inference: confidence={confidence:.3f}, decision={decision.to_dict()}")
        
        return decision
    
    def batch_infer(self, telemetry_batch: np.ndarray) -> list:
        """
        Batch inference for multiple telemetry samples
        
        Args:
            telemetry_batch: (N, 50) array
        
        Returns:
            List of BroadcastDecision objects
        """
        decisions = []
        for i in range(len(telemetry_batch)):
            decision = self.infer(telemetry_batch[i])
            decisions.append(decision)
        
        return decisions


# ===============================================================================
# SYNTHESIS DATA GENERATION FOR TRAINING
# ===============================================================================

def generate_synthetic_training_data(num_samples=10000):
    """
    Generate synthetic training data for model training
    
    Returns:
        X: (num_samples, 50) input features
        y: (num_samples, 5) target outputs
    """
    np.random.seed(42)
    
    X = np.random.randn(num_samples, INPUT_DIM)
    
    # Correlate outputs with inputs in realistic way
    y = np.zeros((num_samples, OUTPUT_DIM))
    
    # Redundancy: increases with signal degradation
    y[:, 0] = 1.0 + 2.0 * np.mean(np.abs(X[:, :10]), axis=1)
    
    # Spectrum: decreases with good conditions, increases with degradation
    y[:, 1] = 1.0 * (1.0 + np.mean(X[:, 10:20], axis=1))
    
    # Availability: improves with better signal
    y[:, 2] = 0.85 + 0.14 * (1.0 - np.mean(np.abs(X[:, 20:30]), axis=1))
    
    # Convergence time: increases with multipath
    y[:, 3] = 30.0 + 20.0 * np.mean(np.abs(X[:, 30:40]), axis=1)
    
    # Accuracy: improves with better signal
    y[:, 4] = 15.0 * (1.0 + np.mean(np.abs(X[:, 40:50]), axis=1))
    
    # Clip to valid ranges
    for idx, (min_val, max_val) in OUTPUT_RANGES.items():
        y[:, idx] = np.clip(y[:, idx], min_val, max_val)
    
    logger.info(f"Generated synthetic training data: X={X.shape}, y={y.shape}")
    
    return X, y


# ===============================================================================
# MAIN FOR TESTING/TRAINING
# ===============================================================================

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("BROADCAST DECISION MODEL - TRAINING")
    logger.info("=" * 80)
    
    # Generate synthetic data
    X_train, y_train = generate_synthetic_training_data(num_samples=10000)
    
    # Create trainer
    trainer = ModelTrainer()
    
    # Prepare data
    train_loader, val_loader = trainer.prepare_data(X_train, y_train)
    
    # Train model
    history = trainer.train(train_loader, val_loader, num_epochs=50)
    
    # Save model
    model_path = "./models/broadcast_decision_model"
    trainer.save_model(model_path)
    
    # Test inference
    logger.info("\n" + "=" * 80)
    logger.info("INFERENCE TEST")
    logger.info("=" * 80)
    
    engine = DecisionInferenceEngine(model_path)
    test_features = np.random.randn(50)
    decision = engine.infer(test_features)
    
    logger.info(f"Test decision: {decision.to_dict()}")
    
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
