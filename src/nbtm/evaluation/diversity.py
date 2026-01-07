"""Topic diversity metrics.

Measures how diverse/distinct topics are from each other.
High diversity indicates topics capture different aspects of the corpus.
"""

from __future__ import annotations

from typing import List, Set, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nbtm.models.base import BaseTopicModel


def compute_topic_diversity(
    model: BaseTopicModel,
    top_n: int = 25,
) -> float:
    """
    Compute topic diversity.

    Topic diversity is the proportion of unique words
    across all topics' top-N words.

    Args:
        model: Trained topic model
        top_n: Number of top words per topic

    Returns:
        Diversity score in [0, 1], higher is better
    """
    topic_words = model.get_all_topic_words(top_n)
    return _compute_diversity(topic_words, top_n)


def _compute_diversity(
    topic_words: List[List[tuple[str, float]]],
    top_n: int,
) -> float:
    """
    Compute diversity from topic-word lists.

    Args:
        topic_words: List of (word, prob) tuples for each topic
        top_n: Number of top words considered

    Returns:
        Diversity score
    """
    all_words: List[str] = []
    for words in topic_words:
        for word, _ in words[:top_n]:
            all_words.append(word)

    if not all_words:
        return 0.0

    unique_words = len(set(all_words))
    total_words = len(all_words)

    return unique_words / total_words


def compute_topic_uniqueness(
    model: BaseTopicModel,
    top_n: int = 10,
) -> List[float]:
    """
    Compute uniqueness score for each topic.

    Uniqueness measures what proportion of a topic's
    top words are unique to that topic.

    Args:
        model: Trained topic model
        top_n: Number of top words per topic

    Returns:
        List of uniqueness scores for each topic
    """
    topic_words = model.get_all_topic_words(top_n)

    # Get all words per topic
    topic_word_sets: List[Set[str]] = [
        {word for word, _ in words[:top_n]}
        for words in topic_words
    ]

    # Compute uniqueness for each topic
    uniqueness = []
    for i, words_i in enumerate(topic_word_sets):
        # Words in other topics
        other_words = set()
        for j, words_j in enumerate(topic_word_sets):
            if i != j:
                other_words.update(words_j)

        # Unique words = words not in any other topic
        unique = words_i - other_words
        score = len(unique) / len(words_i) if words_i else 0.0
        uniqueness.append(score)

    return uniqueness


def compute_topic_overlap(
    model: BaseTopicModel,
    top_n: int = 10,
) -> np.ndarray:
    """
    Compute pairwise topic overlap matrix.

    Overlap is measured as Jaccard similarity between
    top-N word sets.

    Args:
        model: Trained topic model
        top_n: Number of top words per topic

    Returns:
        Overlap matrix of shape (K, K)
    """
    topic_words = model.get_all_topic_words(top_n)

    topic_word_sets = [
        {word for word, _ in words[:top_n]}
        for words in topic_words
    ]

    K = len(topic_word_sets)
    overlap = np.zeros((K, K))

    for i in range(K):
        for j in range(K):
            if i == j:
                overlap[i, j] = 1.0
            else:
                intersection = len(topic_word_sets[i] & topic_word_sets[j])
                union = len(topic_word_sets[i] | topic_word_sets[j])
                overlap[i, j] = intersection / union if union > 0 else 0.0

    return overlap


def compute_inverted_rbo(
    model: BaseTopicModel,
    top_n: int = 10,
    p: float = 0.9,
) -> float:
    """
    Compute inverted Rank-Biased Overlap (RBO).

    RBO measures similarity between ranked lists.
    Inverted RBO measures diversity (1 - average RBO).

    Args:
        model: Trained topic model
        top_n: Number of top words per topic
        p: Persistence parameter (higher = more weight on top ranks)

    Returns:
        Inverted RBO score (higher = more diverse)
    """
    topic_words = model.get_all_topic_words(top_n)
    K = len(topic_words)

    if K < 2:
        return 1.0

    total_rbo = 0.0
    n_pairs = 0

    for i in range(K):
        words_i = [word for word, _ in topic_words[i][:top_n]]
        for j in range(i + 1, K):
            words_j = [word for word, _ in topic_words[j][:top_n]]
            rbo = _compute_rbo(words_i, words_j, p)
            total_rbo += rbo
            n_pairs += 1

    avg_rbo = total_rbo / n_pairs if n_pairs > 0 else 0.0
    return 1.0 - avg_rbo


def _compute_rbo(
    list1: List[str],
    list2: List[str],
    p: float = 0.9,
) -> float:
    """
    Compute Rank-Biased Overlap between two ranked lists.

    Args:
        list1, list2: Ranked word lists
        p: Persistence parameter

    Returns:
        RBO score in [0, 1]
    """
    set1 = set()
    set2 = set()
    agreements = []

    for i, (w1, w2) in enumerate(zip(list1, list2)):
        set1.add(w1)
        set2.add(w2)

        intersection = len(set1 & set2)
        union = len(set1 | set2)
        agreements.append(intersection / union if union > 0 else 0.0)

    # Weight by position
    rbo = 0.0
    for i, a in enumerate(agreements):
        rbo += (p ** i) * a

    # Normalize
    rbo *= (1 - p)
    return rbo
