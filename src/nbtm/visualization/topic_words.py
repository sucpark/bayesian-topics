"""Topic-word visualization tools."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from nbtm.models.base import BaseTopicModel


def plot_topic_words(
    model: BaseTopicModel,
    num_topics: Optional[int] = None,
    top_n: int = 10,
    figsize: tuple[int, int] = (12, 8),
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot bar charts of top words for each topic.

    Args:
        model: Trained topic model
        num_topics: Number of topics to display (None = all)
        top_n: Number of words per topic
        figsize: Figure size
        save_path: Path to save figure (optional)

    Returns:
        matplotlib Figure
    """
    topic_words = model.get_all_topic_words(top_n)

    n_topics = len(topic_words) if num_topics is None else min(num_topics, len(topic_words))

    # Calculate grid dimensions
    n_cols = min(4, n_topics)
    n_rows = (n_topics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    if n_topics == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx in range(n_topics):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        words = [w for w, _ in topic_words[idx][:top_n]]
        probs = [p for _, p in topic_words[idx][:top_n]]

        y_pos = np.arange(len(words))
        ax.barh(y_pos, probs[::-1], color='steelblue', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words[::-1])
        ax.set_title(f'Topic {idx}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Probability')

    # Hide empty subplots
    for idx in range(n_topics, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_topic_heatmap(
    model: BaseTopicModel,
    top_n: int = 20,
    figsize: tuple[int, int] = (14, 10),
    cmap: str = "YlOrRd",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot heatmap of topic-word distributions.

    Args:
        model: Trained topic model
        top_n: Number of top words to show
        figsize: Figure size
        cmap: Colormap
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    topic_word_matrix = model.get_topic_word_distribution()
    vocabulary = model.vocabulary
    n_topics = topic_word_matrix.shape[0]

    # Get top words across all topics
    word_importance = topic_word_matrix.max(axis=0)
    top_word_indices = np.argsort(word_importance)[-top_n:][::-1]

    # Create subset matrix
    subset_matrix = topic_word_matrix[:, top_word_indices]
    subset_words = [vocabulary[i] for i in top_word_indices]

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(subset_matrix, aspect='auto', cmap=cmap)

    ax.set_xticks(range(len(subset_words)))
    ax.set_xticklabels(subset_words, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n_topics))
    ax.set_yticklabels([f'Topic {i}' for i in range(n_topics)])

    ax.set_xlabel('Words', fontsize=11)
    ax.set_ylabel('Topics', fontsize=11)
    ax.set_title('Topic-Word Distribution Heatmap', fontsize=12, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Probability')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
