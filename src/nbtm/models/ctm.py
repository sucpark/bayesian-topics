"""Correlated Topic Model.

This implementation uses the logistic normal distribution
instead of Dirichlet to model topic correlations.

Reference:
    Blei, D. M., & Lafferty, J. D. (2007).
    A correlated topic model of Science. The Annals of Applied Statistics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
from scipy.special import softmax
from tqdm import tqdm

from nbtm.models.base import BaseTopicModel, TopicModelState
from nbtm.models.registry import register_model

if TYPE_CHECKING:
    from nbtm.training.callbacks import Callback


@register_model("ctm")
class CorrelatedTM(BaseTopicModel):
    """
    Correlated Topic Model.

    CTM uses a logistic normal distribution for document-topic
    proportions, allowing topics to be correlated.

    Attributes:
        num_topics: Number of topics K
        beta: Topic-word Dirichlet prior

    Example:
        >>> model = CorrelatedTM(num_topics=10)
        >>> model.fit(documents, num_iterations=100)
        >>> correlation = model.get_topic_correlation()
    """

    def __init__(
        self,
        num_topics: int = 10,
        alpha: float = 0.1,  # Not used in CTM
        beta: float = 0.01,
        random_state: int = 42,
    ) -> None:
        """
        Initialize CTM.

        Args:
            num_topics: Number of topics K
            alpha: Not used (for compatibility)
            beta: Topic-word Dirichlet prior
            random_state: Random seed
        """
        super().__init__(num_topics, alpha, beta, random_state)

        # Logistic normal parameters
        self._mu: Optional[np.ndarray] = None  # Mean (K,)
        self._sigma: Optional[np.ndarray] = None  # Covariance (K, K)

        # Variational parameters
        self._eta: Optional[np.ndarray] = None  # Document-topic (D, K)
        self._lambda: Optional[np.ndarray] = None  # Topic-word (K, V)

        # Documents and vocabulary
        self._documents: List[List[str]] = []
        self._word2idx: Dict[str, int] = {}
        self._vocab_size: int = 0
        self._num_docs: int = 0

    def fit(
        self,
        documents: List[List[str]],
        num_iterations: int = 100,
        callbacks: Optional[List[Callback]] = None,
        verbose: bool = True,
    ) -> CorrelatedTM:
        """
        Fit CTM using variational EM.

        Args:
            documents: List of tokenized documents
            num_iterations: Maximum iterations
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
            iterator = tqdm(iterator, desc="CTM Variational EM")

        for iteration in iterator:
            for cb in callbacks:
                cb.on_iteration_begin(iteration, self)

            # E-step
            self._e_step()

            # M-step
            self._m_step()

            # Log metrics
            if verbose and iteration % 10 == 0:
                self._training_history.append({
                    "iteration": iteration,
                })

            for cb in callbacks:
                cb.on_iteration_end(iteration, self)

        # Finalize
        self._is_fitted = True
        self._state = self._create_state(num_iterations)

        for cb in callbacks:
            cb.on_train_end(self)

        return self

    def _initialize(self, documents: List[List[str]]) -> None:
        """Initialize model parameters."""
        self._documents = documents
        self._num_docs = len(documents)

        # Build vocabulary
        distinct_words = set(word for doc in documents for word in doc)
        self._vocabulary = sorted(list(distinct_words))
        self._word2idx = {w: i for i, w in enumerate(self._vocabulary)}
        self._vocab_size = len(self._vocabulary)

        # Initialize logistic normal parameters
        self._mu = np.zeros(self.num_topics)
        self._sigma = np.eye(self.num_topics)

        # Initialize eta (document-topic in log space)
        self._eta = np.random.randn(self._num_docs, self.num_topics) * 0.1

        # Initialize lambda (topic-word)
        self._lambda = np.random.gamma(
            100, 1 / 100,
            (self.num_topics, self._vocab_size)
        ) + self.beta

    def _e_step(self) -> None:
        """E-step: Update variational parameters for documents."""
        # Get normalized topic-word distribution
        beta = self._lambda / self._lambda.sum(axis=1, keepdims=True)

        for d, doc in enumerate(self._documents):
            if len(doc) == 0:
                continue

            # Gradient ascent for eta_d
            eta_d = self._eta[d].copy()

            for _ in range(5):  # Inner iterations
                # Compute theta from eta
                theta = softmax(eta_d)

                # Gradient of log p(w|eta)
                grad = np.zeros(self.num_topics)
                for word in doc:
                    if word not in self._word2idx:
                        continue
                    v = self._word2idx[word]

                    p_w = np.dot(theta, beta[:, v])
                    if p_w > 0:
                        grad += (beta[:, v] * theta / p_w) - theta

                # Gradient of log p(eta|mu, sigma)
                sigma_inv = np.linalg.inv(self._sigma + 1e-6 * np.eye(self.num_topics))
                grad -= sigma_inv @ (eta_d - self._mu)

                # Update with gradient
                eta_d += 0.01 * grad

            self._eta[d] = eta_d

    def _m_step(self) -> None:
        """M-step: Update mu, sigma, and lambda."""
        # Update mu (mean of eta)
        self._mu = self._eta.mean(axis=0)

        # Update sigma (covariance of eta)
        centered = self._eta - self._mu
        self._sigma = (centered.T @ centered) / self._num_docs
        self._sigma += 1e-6 * np.eye(self.num_topics)  # Regularization

        # Update lambda (topic-word)
        self._lambda = np.ones((self.num_topics, self._vocab_size)) * self.beta

        for d, doc in enumerate(self._documents):
            theta = softmax(self._eta[d])

            for word in doc:
                if word not in self._word2idx:
                    continue
                v = self._word2idx[word]
                self._lambda[:, v] += theta

    def transform(self, documents: List[List[str]]) -> np.ndarray:
        """Infer topic distributions for new documents."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before transform")

        beta = self._lambda / self._lambda.sum(axis=1, keepdims=True)
        theta_new = np.zeros((len(documents), self.num_topics))

        for d, doc in enumerate(documents):
            if len(doc) == 0:
                theta_new[d] = 1.0 / self.num_topics
                continue

            # Initialize eta from prior
            eta_d = self._mu.copy()

            for _ in range(10):
                theta = softmax(eta_d)

                grad = np.zeros(self.num_topics)
                for word in doc:
                    if word not in self._word2idx:
                        continue
                    v = self._word2idx[word]

                    p_w = np.dot(theta, beta[:, v])
                    if p_w > 0:
                        grad += (beta[:, v] * theta / p_w) - theta

                sigma_inv = np.linalg.inv(self._sigma + 1e-6 * np.eye(self.num_topics))
                grad -= sigma_inv @ (eta_d - self._mu)

                eta_d += 0.01 * grad

            theta_new[d] = softmax(eta_d)

        return theta_new

    def get_topic_words(
        self,
        topic_id: int,
        top_n: int = 10,
    ) -> List[tuple[str, float]]:
        """Get top words for a topic."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        beta = self._lambda[topic_id] / self._lambda[topic_id].sum()

        word_probs = [
            (self._vocabulary[v], beta[v])
            for v in range(self._vocab_size)
        ]
        word_probs.sort(key=lambda x: x[1], reverse=True)
        return word_probs[:top_n]

    def get_document_topics(self) -> np.ndarray:
        """Get document-topic distribution matrix."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        theta = np.zeros((self._num_docs, self.num_topics))
        for d in range(self._num_docs):
            theta[d] = softmax(self._eta[d])
        return theta

    def get_topic_word_distribution(self) -> np.ndarray:
        """Get topic-word distribution matrix."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        return self._lambda / self._lambda.sum(axis=1, keepdims=True)

    def get_topic_correlation(self) -> np.ndarray:
        """
        Get topic correlation matrix.

        Returns:
            Correlation matrix of shape (K, K)
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        # Convert covariance to correlation
        std = np.sqrt(np.diag(self._sigma))
        corr = self._sigma / np.outer(std, std)
        return corr

    def log_likelihood(self) -> float:
        """Compute approximate log-likelihood."""
        ll = 0.0
        beta = self._lambda / self._lambda.sum(axis=1, keepdims=True)

        for d, doc in enumerate(self._documents):
            theta = softmax(self._eta[d])

            for word in doc:
                if word not in self._word2idx:
                    continue
                v = self._word2idx[word]
                p_w = np.dot(theta, beta[:, v])
                ll += np.log(p_w + 1e-10)

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
            extra={
                "mu": self._mu.copy(),
                "sigma": self._sigma.copy(),
                "topic_correlation": self.get_topic_correlation(),
            },
        )

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config["uses_logistic_normal"] = True
        return config
