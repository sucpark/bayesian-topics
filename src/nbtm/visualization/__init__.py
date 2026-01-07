"""Visualization tools for topic models."""

from nbtm.visualization.topic_words import (
    plot_topic_words,
    plot_topic_heatmap,
)
from nbtm.visualization.document_topics import (
    plot_document_topics,
    plot_document_topic_bars,
    plot_topic_distribution,
)
from nbtm.visualization.convergence import (
    plot_training_history,
    plot_log_likelihood_curve,
    plot_num_topics_evolution,
)
from nbtm.visualization.comparison import (
    plot_model_comparison,
    plot_radar_comparison,
    plot_training_time_comparison,
)

# Wordcloud is optional
try:
    from nbtm.visualization.wordcloud import (
        plot_topic_wordcloud,
        plot_all_topic_wordclouds,
    )
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

__all__ = [
    # Topic-word
    "plot_topic_words",
    "plot_topic_heatmap",
    # Document-topic
    "plot_document_topics",
    "plot_document_topic_bars",
    "plot_topic_distribution",
    # Convergence
    "plot_training_history",
    "plot_log_likelihood_curve",
    "plot_num_topics_evolution",
    # Comparison
    "plot_model_comparison",
    "plot_radar_comparison",
    "plot_training_time_comparison",
    # Wordcloud (optional)
    "plot_topic_wordcloud",
    "plot_all_topic_wordclouds",
]
