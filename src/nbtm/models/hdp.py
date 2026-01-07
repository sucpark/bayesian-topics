"""Hierarchical Dirichlet Process for Topic Modeling.

This implementation uses the direct assignment Gibbs sampling
algorithm for HDP as described in Teh et al. (2006).

Reference:
    Teh, Y. W., Jordan, M. I., Beal, M. J., & Blei, D. M. (2006).
    Hierarchical Dirichlet Processes. JASA, 101(476), 1566-1581.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from nbtm.models.base import BaseTopicModel, TopicModelState
from nbtm.models.registry import register_model

if TYPE_CHECKING:
    from nbtm.training.callbacks import Callback


@register_model("hdp")
class HierarchicalDP(BaseTopicModel):
    """
    Hierarchical Dirichlet Process for Topic Modeling.

    HDP is a nonparametric Bayesian model that automatically
    infers the number of topics from the data.

    Attributes:
        num_topics: Maximum number of topics (grows as needed)
        gamma: Top-level DP concentration parameter
        alpha0: Document-level DP concentration parameter
        beta: Topic-word Dirichlet prior

    Example:
        >>> model = HierarchicalDP(gamma=1.0, alpha0=1.0)
        >>> model.fit(documents, num_iterations=500)
        >>> print(f"Inferred {model.num_topics} topics")
    """

    def __init__(
        self,
        num_topics: int = 50,  # Initial/max topics
        alpha: float = 0.1,
        beta: float = 0.01,
        gamma: float = 1.0,  # Top-level concentration
        alpha0: float = 1.0,  # Document-level concentration
        random_state: int = 42,
    ) -> None:
        """
        Initialize HDP.

        Args:
            num_topics: Initial maximum topics
            alpha: Not used (for compatibility)
            beta: Topic-word Dirichlet prior
            gamma: Global DP concentration
            alpha0: Document DP concentration
            random_state: Random seed
        """
        super().__init__(num_topics, alpha, beta, random_state)
        self.gamma = gamma
        self.alpha0 = alpha0

        # State variables
        self._documents: List[List[str]] = []
        self._vocabulary: List[str] = []
        self._word2idx: Dict[str, int] = {}
        self._vocab_size: int = 0
        self._num_docs: int = 0

        # Topic assignments and counts
        self._document_topics: List[List[int]] = []
        self._topic_word_counts: Dict[int, Counter] = defaultdict(Counter)
        self._topic_counts: Dict[int, int] = defaultdict(int)
        self._document_topic_counts: List[Counter] = []

        # Active topics
        self._active_topics: set = set()
        self._num_active_topics: int = 0

    @property
    def is_nonparametric(self) -> bool:
        """HDP is a nonparametric model."""
        return True

    def fit(
        self,
        documents: List[List[str]],
        num_iterations: int = 500,
        callbacks: Optional[List[Callback]] = None,
        verbose: bool = True,
    ) -> HierarchicalDP:
        """
        Fit HDP using Gibbs sampling.

        Args:
            documents: List of tokenized documents
            num_iterations: Number of Gibbs sampling iterations
            callbacks: Training callbacks
            verbose: Show progress

        Returns:
            self (fitted model)
        """
        callbacks = callbacks or []
        np.random.seed(self.random_state)

        # Initialize
        self._initialize(documents)

        for cb in callbacks:
            cb.on_train_begin(self)

        iterator = range(num_iterations)
        if verbose:
            iterator = tqdm(iterator, desc="HDP Gibbs Sampling")

        for iteration in iterator:
            for cb in callbacks:
                cb.on_iteration_begin(iteration, self)

            # Gibbs sampling step
            self._gibbs_step()

            # Remove empty topics periodically
            if iteration % 50 == 0:
                self._cleanup_empty_topics()

            # Log metrics
            if verbose and iteration % 50 == 0:
                self._training_history.append({
                    "iteration": iteration,
                    "num_topics": len(self._active_topics),
                })
                if hasattr(iterator, "set_postfix"):
                    iterator.set_postfix({"K": len(self._active_topics)})

            for cb in callbacks:
                cb.on_iteration_end(iteration, self)

        # Cleanup and finalize
        self._cleanup_empty_topics()
        self.num_topics = len(self._active_topics)
        self._is_fitted = True
        self._state = self._create_state(num_iterations)

        for cb in callbacks:
            cb.on_train_end(self)

        return self

    def _initialize(self, documents: List[List[str]]) -> None:
        """Initialize with random topic assignments."""
        self._documents = documents
        self._num_docs = len(documents)

        # Build vocabulary
        distinct_words = set(word for doc in documents for word in doc)
        self._vocabulary = sorted(list(distinct_words))
        self._word2idx = {w: i for i, w in enumerate(self._vocabulary)}
        self._vocab_size = len(self._vocabulary)

        # Initialize counts
        self._document_topic_counts = [Counter() for _ in documents]
        self._topic_word_counts = defaultdict(Counter)
        self._topic_counts = defaultdict(int)

        # Random initialization with limited topics
        initial_k = min(10, self.num_topics)
        self._document_topics = []

        for d, doc in enumerate(documents):
            doc_topics = []
            for word in doc:
                k = np.random.randint(initial_k)
                doc_topics.append(k)

                self._document_topic_counts[d][k] += 1
                self._topic_word_counts[k][word] += 1
                self._topic_counts[k] += 1
                self._active_topics.add(k)

            self._document_topics.append(doc_topics)

        self._num_active_topics = len(self._active_topics)

    def _topic_weight(self, d: int, word: str, k: int) -> float:
        """Compute weight for existing topic k."""
        # P(w|k) with smoothing
        n_kw = self._topic_word_counts[k][word]
        n_k = self._topic_counts[k]
        p_w_k = (n_kw + self.beta) / (n_k + self._vocab_size * self.beta)

        # P(k|d) based on document counts
        n_dk = self._document_topic_counts[d][k]
        p_k_d = n_dk + self.alpha0

        return p_w_k * p_k_d

    def _new_topic_weight(self, word: str) -> float:
        """Compute weight for a new topic."""
        # Prior probability for word in new topic
        p_w_new = 1.0 / self._vocab_size

        # Weight for new topic
        return p_w_new * self.gamma

    def _sample_topic(self, d: int, word: str) -> int:
        """Sample a topic (possibly new)."""
        weights = []
        topic_list = list(self._active_topics)

        # Weights for existing topics
        for k in topic_list:
            w = self._topic_weight(d, word, k)
            weights.append(w)

        # Weight for new topic
        new_weight = self._new_topic_weight(word)
        weights.append(new_weight)

        # Normalize and sample
        total = sum(weights)
        threshold = total * np.random.random()

        cumsum = 0.0
        for i, w in enumerate(weights):
            cumsum += w
            if cumsum >= threshold:
                if i < len(topic_list):
                    return topic_list[i]
                else:
                    # Create new topic
                    new_k = max(self._active_topics) + 1 if self._active_topics else 0
                    self._active_topics.add(new_k)
                    return new_k

        return topic_list[-1] if topic_list else 0

    def _gibbs_step(self) -> None:
        """Perform one Gibbs sampling iteration."""
        for d in range(self._num_docs):
            for i, (word, k) in enumerate(
                zip(self._documents[d], self._document_topics[d])
            ):
                # Remove current assignment
                self._document_topic_counts[d][k] -= 1
                self._topic_word_counts[k][word] -= 1
                self._topic_counts[k] -= 1

                # Sample new topic
                new_k = self._sample_topic(d, word)
                self._document_topics[d][i] = new_k

                # Add new assignment
                self._document_topic_counts[d][new_k] += 1
                self._topic_word_counts[new_k][word] += 1
                self._topic_counts[new_k] += 1
                self._active_topics.add(new_k)

    def _cleanup_empty_topics(self) -> None:
        """Remove topics with no assignments."""
        empty_topics = [
            k for k in self._active_topics
            if self._topic_counts[k] == 0
        ]
        for k in empty_topics:
            self._active_topics.discard(k)
            del self._topic_word_counts[k]
            del self._topic_counts[k]

    def transform(self, documents: List[List[str]]) -> np.ndarray:
        """Infer topic distributions for new documents."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before transform")

        # Map active topics to indices
        topic_map = {k: i for i, k in enumerate(sorted(self._active_topics))}
        n_topics = len(self._active_topics)

        theta = np.zeros((len(documents), n_topics))

        for d, doc in enumerate(documents):
            topic_counts = Counter()

            # Simple inference: assign based on word-topic likelihood
            for word in doc:
                if word not in self._word2idx:
                    continue

                # Find most likely topic
                best_k = None
                best_prob = -1
                for k in self._active_topics:
                    p = (
                        (self._topic_word_counts[k][word] + self.beta) /
                        (self._topic_counts[k] + self._vocab_size * self.beta)
                    )
                    if p > best_prob:
                        best_prob = p
                        best_k = k

                if best_k is not None:
                    topic_counts[best_k] += 1

            # Normalize
            total = sum(topic_counts.values()) + n_topics * self.alpha0
            for k in self._active_topics:
                idx = topic_map[k]
                theta[d, idx] = (topic_counts[k] + self.alpha0) / total

        return theta

    def get_topic_words(
        self,
        topic_id: int,
        top_n: int = 10,
    ) -> List[tuple[str, float]]:
        """Get top words for a topic."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        # Map index to actual topic id
        topic_list = sorted(self._active_topics)
        if topic_id >= len(topic_list):
            return []

        k = topic_list[topic_id]
        n_k = self._topic_counts[k]

        word_probs = []
        for word in self._vocabulary:
            n_kw = self._topic_word_counts[k][word]
            prob = (n_kw + self.beta) / (n_k + self._vocab_size * self.beta)
            word_probs.append((word, prob))

        word_probs.sort(key=lambda x: x[1], reverse=True)
        return word_probs[:top_n]

    def get_document_topics(self) -> np.ndarray:
        """Get document-topic distribution matrix."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        topic_map = {k: i for i, k in enumerate(sorted(self._active_topics))}
        n_topics = len(self._active_topics)

        theta = np.zeros((self._num_docs, n_topics))

        for d in range(self._num_docs):
            doc_len = len(self._documents[d])
            total = doc_len + n_topics * self.alpha0

            for k in self._active_topics:
                idx = topic_map[k]
                n_dk = self._document_topic_counts[d][k]
                theta[d, idx] = (n_dk + self.alpha0) / total

        return theta

    def get_topic_word_distribution(self) -> np.ndarray:
        """Get topic-word distribution matrix."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        topic_list = sorted(self._active_topics)
        n_topics = len(topic_list)

        phi = np.zeros((n_topics, self._vocab_size))

        for i, k in enumerate(topic_list):
            n_k = self._topic_counts[k]
            for v, word in enumerate(self._vocabulary):
                n_kv = self._topic_word_counts[k][word]
                phi[i, v] = (n_kv + self.beta) / (n_k + self._vocab_size * self.beta)

        return phi

    def log_likelihood(self) -> float:
        """Compute log-likelihood of the corpus."""
        ll = 0.0
        topic_list = sorted(self._active_topics)
        n_topics = len(topic_list)

        for d, doc in enumerate(self._documents):
            doc_len = len(doc)
            for word in doc:
                word_prob = 0.0
                for k in topic_list:
                    # P(w|k)
                    n_kw = self._topic_word_counts[k][word]
                    n_k = self._topic_counts[k]
                    p_w_k = (n_kw + self.beta) / (n_k + self._vocab_size * self.beta)

                    # P(k|d)
                    n_dk = self._document_topic_counts[d][k]
                    p_k_d = (n_dk + self.alpha0) / (doc_len + n_topics * self.alpha0)

                    word_prob += p_w_k * p_k_d

                ll += np.log(word_prob + 1e-10)

        return ll

    def _create_state(self, iteration: int) -> TopicModelState:
        """Create model state snapshot."""
        return TopicModelState(
            topic_word_distribution=self.get_topic_word_distribution(),
            document_topic_distribution=self.get_document_topics(),
            num_topics=len(self._active_topics),
            vocabulary=self._vocabulary,
            iteration=iteration,
            log_likelihood=self.log_likelihood(),
            extra={
                "active_topics": list(self._active_topics),
                "gamma": self.gamma,
                "alpha0": self.alpha0,
            },
        )

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config["gamma"] = self.gamma
        config["alpha0"] = self.alpha0
        config["inferred_topics"] = len(self._active_topics)
        return config
