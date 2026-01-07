"""Model comparison visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from nbtm.evaluation.benchmark import BenchmarkResult


def plot_model_comparison(
    results: List[BenchmarkResult],
    metrics: Optional[List[str]] = None,
    figsize: tuple[int, int] = (12, 5),
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot bar chart comparing models across metrics.

    Args:
        results: List of benchmark results
        metrics: Metrics to compare (default: coherence, perplexity, diversity)
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    if metrics is None:
        metrics = ['coherence', 'perplexity', 'diversity']

    model_names = [r.model_name for r in results]
    n_models = len(model_names)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    for ax, metric in zip(axes, metrics):
        values = [getattr(r, metric, None) for r in results]
        valid_values = [v if v is not None else 0 for v in values]

        x = np.arange(n_models)
        bars = ax.bar(x, valid_values, color=colors, edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Model')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')

        # Add value labels
        for bar, val in zip(bars, valid_values):
            if val != 0:
                height = bar.get_height()
                ax.annotate(f'{val:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_radar_comparison(
    results: List[BenchmarkResult],
    figsize: tuple[int, int] = (8, 8),
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot radar chart comparing models.

    Args:
        results: Benchmark results
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    metrics = ['coherence', 'diversity']

    # Normalize values to [0, 1]
    values_dict = {}
    for metric in metrics:
        vals = [getattr(r, metric, 0) or 0 for r in results]
        min_val, max_val = min(vals), max(vals)
        if max_val > min_val:
            normalized = [(v - min_val) / (max_val - min_val) for v in vals]
        else:
            normalized = [0.5] * len(vals)
        values_dict[metric] = normalized

    # Invert perplexity (lower is better)
    perplexity_vals = [r.perplexity or 0 for r in results]
    if max(perplexity_vals) > min(perplexity_vals):
        normalized_perp = [1 - (v - min(perplexity_vals)) / (max(perplexity_vals) - min(perplexity_vals))
                          for v in perplexity_vals]
    else:
        normalized_perp = [0.5] * len(perplexity_vals)
    values_dict['perplexity\n(inverted)'] = normalized_perp
    metrics.append('perplexity\n(inverted)')

    n_metrics = len(metrics)
    n_models = len(results)

    # Angles for radar chart
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    colors = plt.cm.Set1(np.linspace(0, 1, n_models))

    for i, result in enumerate(results):
        values = [values_dict[m][i] for m in metrics]
        values += values[:1]  # Complete the loop

        ax.plot(angles, values, 'o-', linewidth=2, label=result.model_name, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Model Comparison', fontsize=12, fontweight='bold', y=1.08)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_training_time_comparison(
    results: List[BenchmarkResult],
    figsize: tuple[int, int] = (10, 5),
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot training time comparison.

    Args:
        results: Benchmark results
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    model_names = [r.model_name for r in results]
    times = [r.training_time for r in results]

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
    bars = ax.barh(model_names, times, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Training Time (seconds)')
    ax.set_title('Training Time Comparison', fontsize=12, fontweight='bold')

    # Add time labels
    for bar, t in zip(bars, times):
        width = bar.get_width()
        ax.annotate(f'{t:.1f}s',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha='left', va='center', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
