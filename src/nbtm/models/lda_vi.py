"""Variational Inference for Latent Dirichlet Allocation.

This implementation uses mean-field variational inference
as described in Blei, Ng, and Jordan (2003).

Reference:
    Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003).
    Latent Dirichlet Allocation. JMLR, 3, 993-1022.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
from scipy.special import digamma, gammaln
from tqdm import tqdm

from nbtm.models.base import BaseTopicModel, TopicModelState
from nbtm.models.registry import register_model

if TYPE_CHECKING:
    from nbtm.training.callbacks import Callback


@register_model("lda_vi")
class VariationalLDA(BaseTopicModel):
    """
    Latent Dirichlet Allocation with Variational Inference.

    Uses coordinate ascent variational inference to approximate
    the posterior distribution over topics.

    Attributes:
        num_topics: Number of topics K
        alpha: Symmetric Dirichlet prior for document-topic
        beta: Symmetric Dirichlet prior for topic-word

    Example:
        >>> model = VariationalLDA(num_topics=10)
        >>> model.fit(documents, num_iterations=100)
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
        Initialize Variational LDA.

        Args:
            num_topics: Number of topics K
            alpha: Document-topic Dirichlet prior
            beta: Topic-word Dirichlet prior
            random_state: Random seed
        """
        super().__init__(num_topics, alpha, beta, random_state)

        # Variational parameters
        self._gamma: Optional[np.ndarray] = None  # Document-topic (D, K)
        self._phi: Optional[np.ndarray] = None  # Word-topic (D, N_d, K)
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
        convergence_threshold: float = 1e-4,
    ) -> VariationalLDA:
        """
        Fit LDA using variational inference.

        Args:
            documents: List of tokenized documents
            num_iterations: Maximum EM iterations
            callbacks: Optional training callbacks
            verbose: Show progress bar
            convergence_threshold: Convergence threshold for ELBO

        Returns:
            self (fitted model)
        """
        callbacks = callbacks or []
        np.random.seed(self.random_state)

        # Initialize
        self._initialize(documents)

        # Notify callbacks
        for cb in callbacks:
            cb.on_train_begin(self)

        prev_elbo = float("-inf")

        # EM iterations
        iterator = range(num_iterations)
        if verbose:
            iterator = tqdm(iterator, desc="Variational EM")

        for iteration in iterator:
            for cb in callbacks:
                cb.on_iteration_begin(iteration, self)

            # E-step: Update phi and gamma
            self._e_step()

            # M-step: Update lambda
            self._m_step()

            # Compute ELBO
            elbo = self._compute_elbo()
            self._training_history.append({
                "iteration": iteration,
                "elbo": elbo,
            })

            if verbose and hasattr(iterator, "set_postfix"):
                iterator.set_postfix({"ELBO": f"{elbo:.2f}"})

            # Check convergence
            if abs(elbo - prev_elbo) < convergence_threshold:
                if verbose:
                    print(f"\nConverged at iteration {iteration}")
                break
            prev_elbo = elbo

            for cb in callbacks:
                cb.on_iteration_end(iteration, self)

        # Finalize
        self._is_fitted = True
        self._state = self._create_state(iteration)

        for cb in callbacks:
            cb.on_train_end(self)

        return self

    def _initialize(self, documents: List[List[str]]) -> None:
        """Initialize variational parameters."""
        self._documents = documents
        self._num_docs = len(documents)

        # Build vocabulary
        distinct_words = set(word for doc in documents for word in doc)
        self._vocabulary = sorted(list(distinct_words))
        self._word2idx = {w: i for i, w in enumerate(self._vocabulary)}
        self._vocab_size = len(self._vocabulary)

        # Initialize lambda (topic-word) with random + prior
        self._lambda = np.random.gamma(
            100, 1 / 100,
            (self.num_topics, self._vocab_size)
        ) + self.beta

        # Initialize gamma (document-topic)
        self._gamma = np.zeros((self._num_docs, self.num_topics))
        for d in range(self._num_docs):
            self._gamma[d] = self.alpha + len(documents[d]) / self.num_topics

        # Initialize phi for each document
        self._phi_list = []
        for doc in documents:
            n_words = len(doc)
            phi_d = np.ones((n_words, self.num_topics)) / self.num_topics
            self._phi_list.append(phi_d)

    def _e_step(self) -> None:
        """E-step: Update variational parameters phi and gamma."""
        # Precompute expected log theta and log beta
        E_log_beta = digamma(self._lambda) - digamma(
            self._lambda.sum(axis=1, keepdims=True)
        )

        for d, doc in enumerate(self._documents):
            phi_d = self._phi_list[d]
            n_words = len(doc)

            # Update phi
            for n, word in enumerate(doc):
                if word not in self._word2idx:
                    continue
                v = self._word2idx[word]

                # phi_dnk proportional to exp(E[log theta_dk] + E[log beta_kv])
                log_phi = (
                    digamma(self._gamma[d]) -
                    digamma(self._gamma[d].sum()) +
                    E_log_beta[:, v]
                )
                # Normalize
                log_phi -= np.max(log_phi)  # Numerical stability
                phi_d[n] = np.exp(log_phi)
                phi_d[n] /= phi_d[n].sum()

            # Update gamma
            self._gamma[d] = self.alpha + phi_d.sum(axis=0)

    def _m_step(self) -> None:
        """M-step: Update lambda (topic-word sufficient statistics)."""
        self._lambda = np.ones((self.num_topics, self._vocab_size)) * self.beta

        for d, doc in enumerate(self._documents):
            phi_d = self._phi_list[d]
            for n, word in enumerate(doc):
                if word not in self._word2idx:
                    continue
                v = self._word2idx[word]
                self._lambda[:, v] += phi_d[n]

    def _compute_elbo(self) -> float:
        """Compute Evidence Lower Bound (ELBO)."""
        elbo = 0.0

        # E[log p(theta|alpha)]
        for d in range(self._num_docs):
            elbo += (
                gammaln(self.num_topics * self.alpha) -
                self.num_topics * gammaln(self.alpha)
            )
            elbo += np.sum(
                (self.alpha - 1) *
                (digamma(self._gamma[d]) - digamma(self._gamma[d].sum()))
            )

        # E[log p(z|theta)]
        for d, doc in enumerate(self._documents):
            phi_d = self._phi_list[d]
            E_log_theta = digamma(self._gamma[d]) - digamma(self._gamma[d].sum())
            for n in range(len(doc)):
                elbo += np.sum(phi_d[n] * E_log_theta)

        # E[log p(w|z,beta)]
        E_log_beta = digamma(self._lambda) - digamma(
            self._lambda.sum(axis=1, keepdims=True)
        )
        for d, doc in enumerate(self._documents):
            phi_d = self._phi_list[d]
            for n, word in enumerate(doc):
                if word not in self._word2idx:
                    continue
                v = self._word2idx[word]
                elbo += np.sum(phi_d[n] * E_log_beta[:, v])

        # -E[log q(theta)]
        for d in range(self._num_docs):
            elbo -= (
                gammaln(self._gamma[d].sum()) -
                np.sum(gammaln(self._gamma[d]))
            )
            elbo -= np.sum(
                (self._gamma[d] - 1) *
                (digamma(self._gamma[d]) - digamma(self._gamma[d].sum()))
            )

        # -E[log q(z)]
        for d, doc in enumerate(self._documents):
            phi_d = self._phi_list[d]
            for n in range(len(doc)):
                elbo -= np.sum(phi_d[n] * np.log(phi_d[n] + 1e-10))

        return elbo

    def transform(self, documents: List[List[str]]) -> np.ndarray:
        """Infer topic distributions for new documents."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before transform")

        n_docs = len(documents)
        gamma_new = np.zeros((n_docs, self.num_topics))

        # E[log beta]
        E_log_beta = digamma(self._lambda) - digamma(
            self._lambda.sum(axis=1, keepdims=True)
        )

        for d, doc in enumerate(documents):
            # Initialize
            gamma_d = np.ones(self.num_topics) * (
                self.alpha + len(doc) / self.num_topics
            )
            phi_d = np.ones((len(doc), self.num_topics)) / self.num_topics

            # Iterate until convergence
            for _ in range(20):
                # Update phi
                for n, word in enumerate(doc):
                    if word not in self._word2idx:
                        phi_d[n] = 1.0 / self.num_topics
                        continue
                    v = self._word2idx[word]

                    log_phi = (
                        digamma(gamma_d) - digamma(gamma_d.sum()) +
                        E_log_beta[:, v]
                    )
                    log_phi -= np.max(log_phi)
                    phi_d[n] = np.exp(log_phi)
                    phi_d[n] /= phi_d[n].sum()

                # Update gamma
                gamma_d = self.alpha + phi_d.sum(axis=0)

            gamma_new[d] = gamma_d

        # Normalize to get theta
        theta = gamma_new / gamma_new.sum(axis=1, keepdims=True)
        return theta

    def get_topic_words(
        self,
        topic_id: int,
        top_n: int = 10,
    ) -> List[tuple[str, float]]:
        """Get top words for a topic."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        # Normalize lambda to get beta
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

        # Normalize gamma to get theta
        theta = self._gamma / self._gamma.sum(axis=1, keepdims=True)
        return theta

    def get_topic_word_distribution(self) -> np.ndarray:
        """Get topic-word distribution matrix."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        # Normalize lambda to get beta
        beta = self._lambda / self._lambda.sum(axis=1, keepdims=True)
        return beta

    def log_likelihood(self) -> float:
        """Compute approximate log-likelihood (ELBO)."""
        return self._compute_elbo()

    def _create_state(self, iteration: int) -> TopicModelState:
        """Create model state snapshot."""
        return TopicModelState(
            topic_word_distribution=self.get_topic_word_distribution(),
            document_topic_distribution=self.get_document_topics(),
            num_topics=self.num_topics,
            vocabulary=self._vocabulary,
            iteration=iteration,
            log_likelihood=self._compute_elbo(),
            extra={
                "gamma": self._gamma.copy(),
                "lambda": self._lambda.copy(),
            },
        )
