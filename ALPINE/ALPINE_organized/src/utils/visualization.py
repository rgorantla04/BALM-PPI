"""
Visualization utilities for ALPINE experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional


# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_regression(y_true: np.ndarray, y_pred: np.ndarray, metrics: Dict, 
                   title: str, filename: Optional[str] = None) -> None:
    """
    Create and save regression plot.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        metrics: Dictionary with metrics (rmse, pearson, spearman, ci)
        title: Plot title
        filename: Optional path to save the plot
    """
    plt.figure(figsize=(8, 8))
    
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, edgecolor='k', s=50)
    
    # Plot identity line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Identity Line (y=x)')
    
    # Add metrics text box
    metrics_text = (
        f"RMSE: {metrics['rmse']:.4f}\n"
        f"Pearson: {metrics['pearson']:.4f}\n"
        f"Spearman: {metrics['spearman']:.4f}\n"
        f"CI: {metrics['ci']:.4f}"
    )
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
            fc='yellow', alpha=0.5))
    
    plt.title(title, fontsize=16)
    plt.xlabel("Actual pKd", fontsize=14)
    plt.ylabel("Predicted pKd", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to {filename}")
    
    plt.show()


def plot_metrics_comparison(fold_metrics: list, cv_strategy: str, n_folds: int, 
                           filename: Optional[str] = None) -> None:
    """
    Create comparison plot for metrics across folds.
    
    Args:
        fold_metrics: List of metrics dictionaries per fold
        cv_strategy: Name of CV strategy
        n_folds: Number of folds
        filename: Optional path to save the plot
    """
    import pandas as pd
    
    metrics_df = pd.DataFrame(fold_metrics)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"{cv_strategy} CV Metrics ({n_folds}-Fold)", fontsize=16)
    
    metrics_names = ['rmse', 'pearson', 'spearman', 'ci']
    
    for idx, (ax, metric) in enumerate(zip(axes.flat, metrics_names)):
        if metric in metrics_df.columns:
            values = metrics_df[metric].values
            ax.bar(range(1, len(values) + 1), values, alpha=0.7, edgecolor='k')
            ax.axhline(values.mean(), color='r', linestyle='--', label=f'Mean: {values.mean():.4f}')
            ax.set_xlabel('Fold')
            ax.set_ylabel('Score')
            ax.set_title(metric.upper())
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Metrics comparison plot saved to {filename}")
    
    plt.show()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, title: str,
                  filename: Optional[str] = None) -> None:
    """
    Create residual plot.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        title: Plot title
        filename: Optional path to save the plot
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Residual Analysis: {title}", fontsize=14)
    
    # Residuals vs predictions
    axes[0].scatter(y_pred, residuals, alpha=0.6, edgecolor='k', s=50)
    axes[0].axhline(0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel("Predicted pKd")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Predictions")
    axes[0].grid(True, alpha=0.3)
    
    # Distribution of residuals
    axes[1].hist(residuals, bins=20, alpha=0.7, edgecolor='k')
    axes[1].set_xlabel("Residual Value")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Distribution of Residuals")
    axes[1].axvline(0, color='r', linestyle='--', lw=2, label=f'Mean: {residuals.mean():.4f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Residual plot saved to {filename}")
    
    plt.show()
