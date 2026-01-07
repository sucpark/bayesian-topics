"""Convergence visualization for training."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(
    history: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None,
    figsize: tuple[int, int] = (12, 4),
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot training history curves.

    Args:
        history: Training history (list of dicts with iteration and metrics)
        metrics: Metrics to plot (default: all available)
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    if not history:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No training history', ha='center', va='center')
        return fig

    # Extract available metrics
    sample = history[0]
    available_metrics = [k for k in sample.keys() if k != 'iteration']

    if metrics is None:
        metrics = available_metrics
    else:
        metrics = [m for m in metrics if m in available_metrics]

    if not metrics:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No metrics available', ha='center', va='center')
        return fig

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    iterations = [h['iteration'] for h in history]

    for ax, metric in zip(axes, metrics):
        values = [h.get(metric) for h in history]
        values = [v for v in values if v is not None]

        if values:
            ax.plot(iterations[:len(values)], values, 'b-', linewidth=1.5)
            ax.set_xlabel('Iteration')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} over Training')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_log_likelihood_curve(
    history: List[Dict[str, Any]],
    figsize: tuple[int, int] = (10, 5),
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot log-likelihood convergence curve.

    Args:
        history: Training history
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    iterations = [h['iteration'] for h in history]
    ll_values = [h.get('log_likelihood') for h in history]
    ll_values = [v for v in ll_values if v is not None]

    fig, ax = plt.subplots(figsize=figsize)

    if ll_values:
        ax.plot(iterations[:len(ll_values)], ll_values, 'b-', linewidth=2)
        ax.fill_between(iterations[:len(ll_values)], ll_values,
                        alpha=0.2, color='blue')

        # Mark best point
        best_idx = np.argmax(ll_values)
        ax.scatter([iterations[best_idx]], [ll_values[best_idx]],
                   color='red', s=100, zorder=5, label=f'Best: {ll_values[best_idx]:.2f}')

        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Log-Likelihood', fontsize=11)
        ax.set_title('Log-Likelihood Convergence', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No log-likelihood data', ha='center', va='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_num_topics_evolution(
    history: List[Dict[str, Any]],
    figsize: tuple[int, int] = (10, 5),
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot number of topics over training (for HDP).

    Args:
        history: Training history
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    iterations = [h['iteration'] for h in history]
    num_topics = [h.get('num_topics') for h in history]
    num_topics = [v for v in num_topics if v is not None]

    fig, ax = plt.subplots(figsize=figsize)

    if num_topics:
        ax.plot(iterations[:len(num_topics)], num_topics, 'g-', linewidth=2, marker='o')
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Number of Topics', fontsize=11)
        ax.set_title('Topic Count Evolution (HDP)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Final count
        final_k = num_topics[-1]
        ax.axhline(y=final_k, color='red', linestyle='--', alpha=0.7,
                   label=f'Final: {final_k}')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No topic count data', ha='center', va='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
