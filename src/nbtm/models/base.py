"""Abstract base class for topic models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import pickle

import numpy as np

if TYPE_CHECKING:
    from nbtm.training.callbacks import Callback


@dataclass
class TopicModelState:
    """
    State of a topic model after training.

    Contains learned distributions and metadata.
    """

    # Learned distributions
    topic_word_distribution: np.ndarray  # Shape: (K, V), phi
    document_topic_distribution: np.ndarray  # Shape: (D, K), theta

    # Metadata
    num_topics: int
    vocabulary: List[str]
    iteration: int

    # Optional metrics
    log_likelihood: Optional[float] = None
    perplexity: Optional[float] = None

    # Model-specific state
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseTopicModel(ABC):
    """
    Abstract base class for topic models.

    All topic models (LDA, HDP, CTM) inherit from this class.
    Provides a unified interface for training, inference, and evaluation.

    Attributes:
        num_topics: Number of topics (may be learned for nonparametric models)
        alpha: Document-topic prior parameter
        beta: Topic-word prior parameter
        random_state: Random seed for reproducibility
    """

    def __init__(
        self,
        num_topics: int = 10,
        alpha: float = 0.1,
        beta: float = 0.01,
        random_state: int = 42,
    ) -> None:
        """
        Initialize base topic model.

        Args:
            num_topics: Number of topics (initial value for nonparametric models)
            alpha: Document-topic Dirichlet prior
            beta: Topic-word Dirichlet prior
            random_state: Random seed for reproducibility
        """
        self.num_topics = num_topics
        self.alpha = alpha
        self.beta = beta
        self.random_state = random_state

        # Model state
        self._is_fitted = False
        self._state: Optional[TopicModelState] = None
        self._vocabulary: List[str] = []
        self._training_history: List[Dict[str, Any]] = []

    @abstractmethod
    def fit(
        self,
        documents: List[List[str]],
        num_iterations: int = 1000,
        callbacks: Optional[List[Callback]] = None,
    ) -> BaseTopicModel:
        """
        Fit the topic model to the corpus.

        Args:
            documents: List of documents, each document is a list of words
            num_iterations: Number of training iterations
            callbacks: Optional list of training callbacks

        Returns:
            self (fitted model)
        """
        pass

    @abstractmethod
    def transform(self, documents: List[List[str]]) -> np.ndarray:
        """
        Infer topic distribution for new documents.

        Args:
            documents: List of documents to transform

        Returns:
            Document-topic distribution matrix of shape (D, K)
        """
        pass

    def fit_transform(
        self,
        documents: List[List[str]],
        num_iterations: int = 1000,
        callbacks: Optional[List[Callback]] = None,
    ) -> np.ndarray:
        """
        Fit and return document-topic distributions.

        Args:
            documents: Training documents
            num_iterations: Number of iterations
            callbacks: Training callbacks

        Returns:
            Document-topic distribution matrix
        """
        self.fit(documents, num_iterations, callbacks)
        return self.get_document_topics()

    @abstractmethod
    def get_topic_words(
        self,
        topic_id: int,
        top_n: int = 10,
    ) -> List[tuple[str, float]]:
        """
        Get top words for a specific topic.

        Args:
            topic_id: Topic index
            top_n: Number of top words to return

        Returns:
            List of (word, probability) tuples sorted by probability
        """
        pass

    def get_all_topic_words(
        self,
        top_n: int = 10,
    ) -> List[List[tuple[str, float]]]:
        """
        Get top words for all topics.

        Args:
            top_n: Number of top words per topic

        Returns:
            List of topic word lists
        """
        return [self.get_topic_words(k, top_n) for k in range(self.num_topics)]

    @abstractmethod
    def get_document_topics(self) -> np.ndarray:
        """
        Get document-topic distribution matrix.

        Returns:
            Matrix of shape (D, K) where D is number of documents
            and K is number of topics
        """
        pass

    @abstractmethod
    def get_topic_word_distribution(self) -> np.ndarray:
        """
        Get topic-word distribution matrix.

        Returns:
            Matrix of shape (K, V) where K is number of topics
            and V is vocabulary size
        """
        pass

    @abstractmethod
    def log_likelihood(self) -> float:
        """
        Compute log-likelihood of the training corpus.

        Returns:
            Log-likelihood value
        """
        pass

    def perplexity(self, documents: Optional[List[List[str]]] = None) -> float:
        """
        Compute perplexity on documents.

        Args:
            documents: Documents to compute perplexity on
                      (uses training documents if None)

        Returns:
            Perplexity value (lower is better)
        """
        if documents is None:
            ll = self.log_likelihood()
            total_words = sum(len(doc) for doc in self._documents)
        else:
            # For new documents, need to infer topic distributions first
            raise NotImplementedError(
                "Perplexity on new documents not implemented in base class"
            )

        return np.exp(-ll / total_words)

    def get_state(self) -> TopicModelState:
        """
        Get current model state.

        Returns:
            TopicModelState with learned distributions

        Raises:
            RuntimeError: If model has not been fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet")
        return self._state

    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration.

        Returns:
            Dictionary of configuration parameters
        """
        return {
            "model_type": self.__class__.__name__,
            "num_topics": self.num_topics,
            "alpha": self.alpha,
            "beta": self.beta,
            "random_state": self.random_state,
        }

    @property
    def is_fitted(self) -> bool:
        """Check if model has been fitted."""
        return self._is_fitted

    @property
    def is_nonparametric(self) -> bool:
        """
        Whether this is a nonparametric model.

        Nonparametric models (like HDP) learn the number of topics.
        Override in subclass if True.
        """
        return False

    @property
    def vocabulary(self) -> List[str]:
        """Get vocabulary."""
        return self._vocabulary

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self._vocabulary)

    @property
    def training_history(self) -> List[Dict[str, Any]]:
        """Get training history."""
        return self._training_history

    def save(self, path: Path | str) -> None:
        """
        Save model to file.

        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "config": self.get_config(),
            "state": self._state,
            "vocabulary": self._vocabulary,
            "is_fitted": self._is_fitted,
            "training_history": self._training_history,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: Path | str) -> BaseTopicModel:
        """
        Load model from file.

        Args:
            path: Path to load model from

        Returns:
            Loaded model instance
        """
        path = Path(path)

        with open(path, "rb") as f:
            state = pickle.load(f)

        config = state["config"]
        model = cls(
            num_topics=config["num_topics"],
            alpha=config["alpha"],
            beta=config["beta"],
            random_state=config["random_state"],
        )

        model._state = state["state"]
        model._vocabulary = state["vocabulary"]
        model._is_fitted = state["is_fitted"]
        model._training_history = state["training_history"]

        return model

    def print_topics(self, top_n: int = 10) -> None:
        """
        Print all topics with top words.

        Args:
            top_n: Number of words per topic
        """
        for k in range(self.num_topics):
            words = self.get_topic_words(k, top_n)
            word_str = ", ".join(f"{w}({p:.3f})" for w, p in words)
            print(f"Topic {k}: {word_str}")

    def __repr__(self) -> str:
        """String representation."""
        fitted_str = "fitted" if self._is_fitted else "not fitted"
        return (
            f"{self.__class__.__name__}("
            f"num_topics={self.num_topics}, "
            f"alpha={self.alpha}, "
            f"beta={self.beta}, "
            f"{fitted_str})"
        )
