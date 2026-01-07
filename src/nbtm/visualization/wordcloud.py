"""Word cloud visualization for topics."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

if TYPE_CHECKING:
    from nbtm.models.base import BaseTopicModel


def plot_topic_wordcloud(
    model: BaseTopicModel,
    topic_id: int,
    top_n: int = 50,
    figsize: tuple[int, int] = (10, 6),
    background_color: str = "white",
    colormap: str = "viridis",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot word cloud for a single topic.

    Args:
        model: Trained topic model
        topic_id: Topic index
        top_n: Number of words to include
        figsize: Figure size
        background_color: Background color
        colormap: Color map for words
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    if not WORDCLOUD_AVAILABLE:
        raise ImportError("wordcloud package required. Install with: pip install wordcloud")

    topic_words = model.get_topic_words(topic_id, top_n)

    # Create word frequency dictionary
    word_freq = {word: prob for word, prob in topic_words}

    # Generate word cloud
    wc = WordCloud(
        width=800,
        height=400,
        background_color=background_color,
        colormap=colormap,
        max_words=top_n,
        prefer_horizontal=0.7,
    ).generate_from_frequencies(word_freq)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Topic {topic_id}', fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_all_topic_wordclouds(
    model: BaseTopicModel,
    num_topics: Optional[int] = None,
    top_n: int = 30,
    figsize: tuple[int, int] = (16, 12),
    background_color: str = "white",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot word clouds for all topics in a grid.

    Args:
        model: Trained topic model
        num_topics: Number of topics to show (None = all)
        top_n: Words per topic
        figsize: Figure size
        background_color: Background color
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    if not WORDCLOUD_AVAILABLE:
        raise ImportError("wordcloud package required. Install with: pip install wordcloud")

    topic_words = model.get_all_topic_words(top_n)
    n_topics = len(topic_words) if num_topics is None else min(num_topics, len(topic_words))

    # Grid layout
    n_cols = min(4, n_topics)
    n_rows = (n_topics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    if n_topics == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
                 'Blues', 'Greens', 'Reds', 'Purples', 'Oranges']

    for idx in range(n_topics):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        word_freq = {word: prob for word, prob in topic_words[idx]}

        wc = WordCloud(
            width=400,
            height=300,
            background_color=background_color,
            colormap=colormaps[idx % len(colormaps)],
            max_words=top_n,
        ).generate_from_frequencies(word_freq)

        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'Topic {idx}', fontsize=10, fontweight='bold')

    # Hide empty subplots
    for idx in range(n_topics, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
