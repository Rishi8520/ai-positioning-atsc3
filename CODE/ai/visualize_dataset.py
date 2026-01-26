"""
Dataset Visualization Script
Creates plots to understand dataset distribution
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Dataset directory
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent.parent / "DATA"
PLOTS_DIR = DATA_DIR / "Plots"

def visualize_dataset():
    """Create visualizations of the dataset"""
    
    print("Loading dataset...")
    X_train = np.load(DATA_DIR / 'X_train.npy')
    y_train = np.load(DATA_DIR / 'y_train.npy')
    
    # Create plots directory
    PLOTS_DIR.mkdir(exist_ok=True)
    
    target_names = ['Redundancy Ratio', 'Spectrum (Mbps)', 'Availability (%)', 
                    'Convergence Time (s)', 'Accuracy HPE (cm)']
    
    # Plot 1: Target distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, name in enumerate(target_names):
        axes[i].hist(y_train[:, i], bins=50, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'{name}\nMean: {y_train[:, i].mean():.2f}, Std: {y_train[:, i].std():.2f}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(alpha=0.3)
    
    axes[-1].axis('off')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'target_distributions.png', dpi=150)
    print(f"✓ Saved: {PLOTS_DIR / 'target_distributions.png'}")
    
    # Plot 2: Feature correlations with targets
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # Key features to plot
    feature_indices = [0, 30, 40, 44, 47]  # signal_strength[0], snr[0], urban_density, nlos_prob, speed
    feature_names = ['Signal Strength', 'SNR', 'Urban Density', 'NLOS Prob', 'Vehicle Speed']
    
    for i, (feat_idx, feat_name) in enumerate(zip(feature_indices, feature_names)):
        # Plot correlation with accuracy (target 4)
        axes[i].scatter(X_train[:, feat_idx], y_train[:, 4], alpha=0.1, s=1)
        axes[i].set_xlabel(feat_name)
        axes[i].set_ylabel('Accuracy HPE (cm)')
        axes[i].set_title(f'{feat_name} vs Accuracy')
        axes[i].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'feature_correlations.png', dpi=150)
    print(f"✓ Saved: {PLOTS_DIR / 'feature_correlations.png'}")
    
    # Plot 3: Target correlations
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    # Redundancy vs Accuracy
    axes[0].scatter(y_train[:, 0], y_train[:, 4], alpha=0.3, s=2)
    axes[0].set_xlabel('Redundancy Ratio')
    axes[0].set_ylabel('Accuracy HPE (cm)')
    axes[0].set_title('Redundancy vs Accuracy')
    axes[0].grid(alpha=0.3)
    
    # Spectrum vs Redundancy
    axes[1].scatter(y_train[:, 1], y_train[:, 0], alpha=0.3, s=2)
    axes[1].set_xlabel('Spectrum (Mbps)')
    axes[1].set_ylabel('Redundancy Ratio')
    axes[1].set_title('Spectrum vs Redundancy')
    axes[1].grid(alpha=0.3)
    
    # Availability vs Convergence
    axes[2].scatter(y_train[:, 2], y_train[:, 3], alpha=0.3, s=2)
    axes[2].set_xlabel('Availability (%)')
    axes[2].set_ylabel('Convergence Time (s)')
    axes[2].set_title('Availability vs Convergence')
    axes[2].grid(alpha=0.3)
    
    # Convergence vs Accuracy
    axes[3].scatter(y_train[:, 3], y_train[:, 4], alpha=0.3, s=2)
    axes[3].set_xlabel('Convergence Time (s)')
    axes[3].set_ylabel('Accuracy HPE (cm)')
    axes[3].set_title('Convergence vs Accuracy')
    axes[3].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'target_correlations.png', dpi=150)
    print(f"✓ Saved: {PLOTS_DIR / 'target_correlations.png'}")
    
    print(f"\n✅ All plots saved to: {PLOTS_DIR}")

if __name__ == "__main__":
    visualize_dataset()