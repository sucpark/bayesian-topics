"""Perplexity computation for topic models.

Perplexity measures how well the model predicts held-out data.
Lower perplexity indicates better generalization.
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nbtm.models.base import BaseTopicModel


def compute_perplexity(
    model: BaseTopicModel,
    documents: Optional[List[List[str]]] = None,
) -> float:
    """
    Compute perplexity of the model.

    Perplexity = exp(-log_likelihood / num_words)

    Args:
        model: Trained topic model
        documents: Documents to compute perplexity on
                  (uses training documents if None)

    Returns:
        Perplexity value (lower is better)
    """
    if documents is None:
        # Use training corpus
        ll = model.log_likelihood()
        total_words = sum(len(doc) for doc in model._documents)
    else:
        # Compute on new documents
        ll = compute_log_likelihood(model, documents)
        total_words = sum(len(doc) for doc in documents)

    if total_words == 0:
        return float("inf")

    return np.exp(-ll / total_words)


def compute_log_likelihood(
    model: BaseTopicModel,
    documents: List[List[str]],
) -> float:
    """
    Compute log-likelihood on given documents.

    Args:
        model: Trained topic model
        documents: Documents to evaluate

    Returns:
        Log-likelihood value
    """
    if not model.is_fitted:
        raise RuntimeError("Model must be fitted first")

    # Get topic-word distribution
    phi = model.get_topic_word_distribution()  # (K, V)
    vocab = model.vocabulary
    word2idx = {w: i for i, w in enumerate(vocab)}

    # Infer document-topic distributions
    theta = model.transform(documents)  # (D, K)

    ll = 0.0
    for d, doc in enumerate(documents):
        for word in doc:
            if word not in word2idx:
                continue

            v = word2idx[word]

            # P(w) = sum_k P(w|k) * P(k|d)
            word_prob = np.dot(theta[d], phi[:, v])
            ll += np.log(word_prob + 1e-10)

    return ll


def compute_held_out_perplexity(
    model: BaseTopicModel,
    documents: List[List[str]],
    held_out_ratio: float = 0.5,
) -> float:
    """
    Compute held-out perplexity.

    Splits each document into observed and held-out portions,
    infers topic distribution from observed part, and
    computes perplexity on held-out part.

    Args:
        model: Trained topic model
        documents: Documents to evaluate
        held_out_ratio: Proportion of words to hold out

    Returns:
        Held-out perplexity
    """
    if not model.is_fitted:
        raise RuntimeError("Model must be fitted first")

    rng = np.random.RandomState(42)

    observed_docs = []
    held_out_docs = []

    for doc in documents:
        n = len(doc)
        n_held_out = max(1, int(n * held_out_ratio))
        indices = rng.permutation(n)

        held_out_idx = set(indices[:n_held_out])
        observed = [w for i, w in enumerate(doc) if i not in held_out_idx]
        held_out = [w for i, w in enumerate(doc) if i in held_out_idx]

        observed_docs.append(observed)
        held_out_docs.append(held_out)

    # Infer topic distribution from observed words
    theta = model.transform(observed_docs)

    # Compute log-likelihood on held-out words
    phi = model.get_topic_word_distribution()
    vocab = model.vocabulary
    word2idx = {w: i for i, w in enumerate(vocab)}

    ll = 0.0
    total_words = 0

    for d, doc in enumerate(held_out_docs):
        for word in doc:
            if word not in word2idx:
                continue

            v = word2idx[word]
            word_prob = np.dot(theta[d], phi[:, v])
            ll += np.log(word_prob + 1e-10)
            total_words += 1

    if total_words == 0:
        return float("inf")

    return np.exp(-ll / total_words)


class PerplexityTracker:
    """
    Track perplexity across training iterations.

    Useful for monitoring convergence and detecting overfitting.
    """

    def __init__(self) -> None:
        """Initialize tracker."""
        self.iterations: List[int] = []
        self.train_perplexity: List[float] = []
        self.val_perplexity: List[float] = []

    def add(
        self,
        iteration: int,
        train: float,
        val: Optional[float] = None,
    ) -> None:
        """Add perplexity values for an iteration."""
        self.iterations.append(iteration)
        self.train_perplexity.append(train)
        if val is not None:
            self.val_perplexity.append(val)

    @property
    def best_train(self) -> Optional[float]:
        """Get best (lowest) training perplexity."""
        if not self.train_perplexity:
            return None
        return min(self.train_perplexity)

    @property
    def best_val(self) -> Optional[float]:
        """Get best (lowest) validation perplexity."""
        if not self.val_perplexity:
            return None
        return min(self.val_perplexity)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "iterations": self.iterations,
            "train_perplexity": self.train_perplexity,
            "val_perplexity": self.val_perplexity,
        }
