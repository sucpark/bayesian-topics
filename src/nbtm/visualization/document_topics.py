"""Document-topic visualization tools."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if TYPE_CHECKING:
    from nbtm.models.base import BaseTopicModel


def plot_document_topics(
    model: BaseTopicModel,
    num_docs: Optional[int] = None,
    figsize: tuple[int, int] = (12, 8),
    cmap: str = "Blues",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot heatmap of document-topic distributions.

    Args:
        model: Trained topic model
        num_docs: Number of documents to show (None = all)
        figsize: Figure size
        cmap: Colormap
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    theta = model.get_document_topics()

    if num_docs is not None:
        theta = theta[:num_docs]

    n_docs, n_topics = theta.shape

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(theta, aspect='auto', cmap=cmap)

    ax.set_xlabel('Topics', fontsize=11)
    ax.set_ylabel('Documents', fontsize=11)
    ax.set_xticks(range(n_topics))
    ax.set_xticklabels([f'T{i}' for i in range(n_topics)])

    if n_docs <= 30:
        ax.set_yticks(range(n_docs))
        ax.set_yticklabels([f'Doc {i}' for i in range(n_docs)])

    ax.set_title('Document-Topic Distribution', fontsize=12, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Probability')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_document_topic_bars(
    model: BaseTopicModel,
    doc_indices: List[int],
    figsize: tuple[int, int] = (12, 6),
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot stacked bar chart of topic proportions for selected documents.

    Args:
        model: Trained topic model
        doc_indices: Document indices to plot
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    theta = model.get_document_topics()
    n_topics = theta.shape[1]

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(doc_indices))
    width = 0.8

    colors = plt.cm.Set3(np.linspace(0, 1, n_topics))

    bottom = np.zeros(len(doc_indices))
    for k in range(n_topics):
        values = theta[doc_indices, k]
        ax.bar(x, values, width, bottom=bottom, label=f'Topic {k}',
               color=colors[k], edgecolor='white', linewidth=0.5)
        bottom += values

    ax.set_xlabel('Document', fontsize=11)
    ax.set_ylabel('Topic Proportion', fontsize=11)
    ax.set_title('Topic Distribution per Document', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Doc {i}' for i in doc_indices])
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_topic_distribution(
    model: BaseTopicModel,
    figsize: tuple[int, int] = (10, 6),
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot the overall distribution of topics across the corpus.

    Args:
        model: Trained topic model
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    theta = model.get_document_topics()

    # Average topic distribution
    avg_topic = theta.mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Bar plot
    ax1 = axes[0]
    x = np.arange(len(avg_topic))
    ax1.bar(x, avg_topic, color='steelblue', alpha=0.8)
    ax1.set_xlabel('Topic', fontsize=11)
    ax1.set_ylabel('Average Proportion', fontsize=11)
    ax1.set_title('Average Topic Distribution', fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'T{i}' for i in range(len(avg_topic))])

    # Box plot
    ax2 = axes[1]
    ax2.boxplot([theta[:, k] for k in range(theta.shape[1])],
                labels=[f'T{k}' for k in range(theta.shape[1])])
    ax2.set_xlabel('Topic', fontsize=11)
    ax2.set_ylabel('Proportion', fontsize=11)
    ax2.set_title('Topic Distribution Spread', fontsize=11, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
