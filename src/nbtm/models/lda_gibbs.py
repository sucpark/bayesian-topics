"""Gibbs Sampling for Latent Dirichlet Allocation.

This implementation is based on the collapsed Gibbs sampling algorithm
for LDA as described in Griffiths and Steyvers (2004).

Reference:
    Griffiths, T. L., & Steyvers, M. (2004). Finding scientific topics.
    Proceedings of the National Academy of Sciences, 101(suppl 1), 5228-5235.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from nbtm.models.base import BaseTopicModel, TopicModelState
from nbtm.models.registry import register_model

if TYPE_CHECKING:
    from nbtm.training.callbacks import Callback


@register_model("lda_gibbs")
class GibbsLDA(BaseTopicModel):
    """
    Latent Dirichlet Allocation with Collapsed Gibbs Sampling.

    This implementation uses collapsed Gibbs sampling where the
    topic-word and document-topic distributions are integrated out.

    Attributes:
        num_topics: Number of topics K
        alpha: Symmetric Dirichlet prior for document-topic distribution
        beta: Symmetric Dirichlet prior for topic-word distribution

    Example:
        >>> model = GibbsLDA(num_topics=10, alpha=0.1, beta=0.01)
        >>> model.fit(documents, num_iterations=1000)
        >>> topics = model.get_all_topic_words(top_n=10)
    """

    def __init__(
        self,
        num_topics: int = 10,
        alpha: float = 0.1,
        beta: float = 0.01,
        random_state: int = 42,
    ) -> None:
        """
        Initialize Gibbs Sampling LDA.

        Args:
            num_topics: Number of topics K
            alpha: Document-topic Dirichlet prior (symmetric)
            beta: Topic-word Dirichlet prior (symmetric)
            random_state: Random seed for reproducibility
        """
        super().__init__(num_topics, alpha, beta, random_state)

        # Count matrices (initialized during fit)
        self._document_topic_counts: List[Counter] = []
        self._topic_word_counts: List[Counter] = []
        self._topic_counts: List[int] = []
        self._document_lengths: List[int] = []

        # Topic assignments for each word
        self._document_topics: List[List[int]] = []

        # Documents and vocabulary
        self._documents: List[List[str]] = []
        self._vocab_size: int = 0
        self._num_docs: int = 0

    def fit(
        self,
        documents: List[List[str]],
        num_iterations: int = 1000,
        callbacks: Optional[List[Callback]] = None,
        verbose: bool = True,
    ) -> GibbsLDA:
        """
        Fit LDA model using Gibbs sampling.

        Args:
            documents: List of tokenized documents
            num_iterations: Number of Gibbs sampling iterations
            callbacks: Optional training callbacks
            verbose: Show progress bar

        Returns:
            self (fitted model)
        """
        callbacks = callbacks or []

        # Initialize
        self._initialize(documents)
        np.random.seed(self.random_state)

        # Notify callbacks
        for cb in callbacks:
            cb.on_train_begin(self)

        # Gibbs sampling iterations
        iterator = range(num_iterations)
        if verbose:
            iterator = tqdm(iterator, desc="Gibbs Sampling")

        for iteration in iterator:
            for cb in callbacks:
                cb.on_iteration_begin(iteration, self)

            # Perform one Gibbs sampling step
            self._gibbs_step()

            # Log metrics
            if verbose and iteration % 100 == 0:
                ll = self.log_likelihood()
                self._training_history.append({
                    "iteration": iteration,
                    "log_likelihood": ll,
                })
                if hasattr(iterator, "set_postfix"):
                    iterator.set_postfix({"ll": f"{ll:.2f}"})

            for cb in callbacks:
                cb.on_iteration_end(iteration, self)

        # Finalize
        self._is_fitted = True
        self._state = self._create_state(num_iterations)

        for cb in callbacks:
            cb.on_train_end(self)

        return self

    def _initialize(self, documents: List[List[str]]) -> None:
        """
        Initialize count matrices and random topic assignments.

        Args:
            documents: Tokenized documents
        """
        self._documents = documents
        self._num_docs = len(documents)

        # Build vocabulary
        distinct_words = set(word for doc in documents for word in doc)
        self._vocabulary = sorted(list(distinct_words))
        self._vocab_size = len(self._vocabulary)

        # Initialize counts
        self._document_topic_counts = [Counter() for _ in documents]
        self._topic_word_counts = [Counter() for _ in range(self.num_topics)]
        self._topic_counts = [0] * self.num_topics
        self._document_lengths = list(map(len, documents))

        # Random initialization of topic assignments
        self._document_topics = [
            [np.random.randint(self.num_topics) for _ in doc]
            for doc in documents
        ]

        # Count initial assignments
        for d in range(self._num_docs):
            for word, topic in zip(documents[d], self._document_topics[d]):
                self._document_topic_counts[d][topic] += 1
                self._topic_word_counts[topic][word] += 1
                self._topic_counts[topic] += 1

    def _p_topic_given_document(self, topic: int, d: int) -> float:
        """
        Compute P(topic | document) with Dirichlet smoothing.

        Args:
            topic: Topic index
            d: Document index

        Returns:
            Probability of topic given document
        """
        return (
            (self._document_topic_counts[d][topic] + self.alpha) /
            (self._document_lengths[d] + self.num_topics * self.alpha)
        )

    def _p_word_given_topic(self, word: str, topic: int) -> float:
        """
        Compute P(word | topic) with Dirichlet smoothing.

        Args:
            word: Word string
            topic: Topic index

        Returns:
            Probability of word given topic
        """
        return (
            (self._topic_word_counts[topic][word] + self.beta) /
            (self._topic_counts[topic] + self._vocab_size * self.beta)
        )

    def _topic_weight(self, d: int, word: str, k: int) -> float:
        """
        Compute weight for sampling topic k.

        Args:
            d: Document index
            word: Word string
            k: Topic index

        Returns:
            Unnormalized weight for topic k
        """
        return (
            self._p_word_given_topic(word, k) *
            self._p_topic_given_document(k, d)
        )

    def _sample_topic(self, d: int, word: str) -> int:
        """
        Sample a new topic for a word.

        Args:
            d: Document index
            word: Word string

        Returns:
            Sampled topic index
        """
        weights = [self._topic_weight(d, word, k) for k in range(self.num_topics)]
        total = sum(weights)

        # Sample from categorical distribution
        threshold = total * np.random.random()
        cumsum = 0.0
        for k, w in enumerate(weights):
            cumsum += w
            if cumsum >= threshold:
                return k

        return self.num_topics - 1

    def _gibbs_step(self) -> None:
        """Perform one complete Gibbs sampling iteration."""
        for d in range(self._num_docs):
            for i, (word, topic) in enumerate(
                zip(self._documents[d], self._document_topics[d])
            ):
                # Remove current word-topic assignment from counts
                self._document_topic_counts[d][topic] -= 1
                self._topic_word_counts[topic][word] -= 1
                self._topic_counts[topic] -= 1
                self._document_lengths[d] -= 1

                # Sample new topic
                new_topic = self._sample_topic(d, word)
                self._document_topics[d][i] = new_topic

                # Add new word-topic assignment to counts
                self._document_topic_counts[d][new_topic] += 1
                self._topic_word_counts[new_topic][word] += 1
                self._topic_counts[new_topic] += 1
                self._document_lengths[d] += 1

    def transform(self, documents: List[List[str]]) -> np.ndarray:
        """
        Infer topic distributions for new documents.

        Uses Gibbs sampling with fixed topic-word distributions
        to infer document-topic distributions.

        Args:
            documents: New documents to transform

        Returns:
            Document-topic distribution matrix of shape (D, K)
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before transform")

        # Run inference iterations for new documents
        n_docs = len(documents)
        n_iter = 100  # Inference iterations

        # Initialize
        doc_topic_counts = [Counter() for _ in documents]
        doc_topics = [
            [np.random.randint(self.num_topics) for _ in doc]
            for doc in documents
        ]
        doc_lengths = list(map(len, documents))

        # Count initial assignments
        for d in range(n_docs):
            for word, topic in zip(documents[d], doc_topics[d]):
                doc_topic_counts[d][topic] += 1

        # Inference iterations (with fixed phi)
        for _ in range(n_iter):
            for d in range(n_docs):
                for i, (word, topic) in enumerate(zip(documents[d], doc_topics[d])):
                    doc_topic_counts[d][topic] -= 1
                    doc_lengths[d] -= 1

                    # Sample using learned topic-word distribution
                    weights = []
                    for k in range(self.num_topics):
                        p_w_k = self._p_word_given_topic(word, k)
                        p_k_d = (
                            (doc_topic_counts[d][k] + self.alpha) /
                            (doc_lengths[d] + self.num_topics * self.alpha)
                        )
                        weights.append(p_w_k * p_k_d)

                    total = sum(weights)
                    threshold = total * np.random.random()
                    cumsum = 0.0
                    new_topic = self.num_topics - 1
                    for k, w in enumerate(weights):
                        cumsum += w
                        if cumsum >= threshold:
                            new_topic = k
                            break

                    doc_topics[d][i] = new_topic
                    doc_topic_counts[d][new_topic] += 1
                    doc_lengths[d] += 1

        # Compute theta (document-topic distribution)
        theta = np.zeros((n_docs, self.num_topics))
        for d in range(n_docs):
            for k in range(self.num_topics):
                theta[d, k] = (
                    (doc_topic_counts[d][k] + self.alpha) /
                    (doc_lengths[d] + self.num_topics * self.alpha)
                )

        return theta

    def get_topic_words(
        self,
        topic_id: int,
        top_n: int = 10,
    ) -> List[tuple[str, float]]:
        """
        Get top words for a specific topic.

        Args:
            topic_id: Topic index
            top_n: Number of top words

        Returns:
            List of (word, probability) tuples
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        word_probs = [
            (word, self._p_word_given_topic(word, topic_id))
            for word in self._vocabulary
        ]
        word_probs.sort(key=lambda x: x[1], reverse=True)
        return word_probs[:top_n]

    def get_document_topics(self) -> np.ndarray:
        """
        Get document-topic distribution matrix.

        Returns:
            Matrix theta of shape (D, K)
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        theta = np.zeros((self._num_docs, self.num_topics))
        for d in range(self._num_docs):
            for k in range(self.num_topics):
                theta[d, k] = self._p_topic_given_document(k, d)
        return theta

    def get_topic_word_distribution(self) -> np.ndarray:
        """
        Get topic-word distribution matrix.

        Returns:
            Matrix phi of shape (K, V)
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        phi = np.zeros((self.num_topics, self._vocab_size))
        for k in range(self.num_topics):
            for v, word in enumerate(self._vocabulary):
                phi[k, v] = self._p_word_given_topic(word, k)
        return phi

    def log_likelihood(self) -> float:
        """
        Compute log-likelihood of the corpus.

        Returns:
            Log-likelihood value
        """
        ll = 0.0
        for d, doc in enumerate(self._documents):
            for word in doc:
                word_prob = sum(
                    self._p_word_given_topic(word, k) *
                    self._p_topic_given_document(k, d)
                    for k in range(self.num_topics)
                )
                ll += np.log(word_prob + 1e-10)
        return ll

    def _create_state(self, iteration: int) -> TopicModelState:
        """Create model state snapshot."""
        return TopicModelState(
            topic_word_distribution=self.get_topic_word_distribution(),
            document_topic_distribution=self.get_document_topics(),
            num_topics=self.num_topics,
            vocabulary=self._vocabulary,
            iteration=iteration,
            log_likelihood=self.log_likelihood(),
        )

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config["vocab_size"] = self._vocab_size
        config["num_docs"] = self._num_docs
        return config
