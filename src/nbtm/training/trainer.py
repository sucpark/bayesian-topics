"""Trainer class for topic models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from nbtm.config import Config
from nbtm.data import Corpus
from nbtm.models import create_model
from nbtm.training.callbacks import (
    Callback,
    CallbackList,
    EarlyStopping,
    HistoryLogger,
    ModelCheckpoint,
    ProgressLogger,
)

if TYPE_CHECKING:
    from nbtm.models.base import BaseTopicModel


class Trainer:
    """
    Trainer for topic models.

    Provides a high-level interface for training topic models
    with callbacks, checkpointing, and logging.

    Example:
        >>> config = Config.from_yaml("configs/default.yaml")
        >>> trainer = Trainer(config)
        >>> model = trainer.fit(corpus)
    """

    def __init__(
        self,
        config: Config,
        callbacks: Optional[List[Callback]] = None,
    ) -> None:
        """
        Initialize trainer.

        Args:
            config: Training configuration
            callbacks: Additional callbacks
        """
        self.config = config
        self.callbacks = CallbackList(callbacks)

        # Setup directories
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup default callbacks
        self._setup_default_callbacks()

    def _setup_default_callbacks(self) -> None:
        """Setup default callbacks from config."""
        # Progress logger
        if self.config.verbose:
            self.callbacks.append(
                ProgressLogger(log_every=self.config.training.log_every)
            )

        # History logger
        history_path = self.output_dir / "history.json"
        self.callbacks.append(
            HistoryLogger(filepath=history_path, log_every=self.config.training.log_every)
        )

        # Model checkpoint
        checkpoint_path = self.output_dir / "best_model.pkl"
        self.callbacks.append(
            ModelCheckpoint(
                filepath=checkpoint_path,
                save_best_only=True,
                save_every=self.config.training.save_every,
            )
        )

        # Early stopping
        if self.config.training.early_stopping:
            self.callbacks.append(
                EarlyStopping(patience=self.config.training.patience)
            )

    def fit(
        self,
        corpus: Corpus,
        model: Optional[BaseTopicModel] = None,
    ) -> BaseTopicModel:
        """
        Train a topic model on the corpus.

        Args:
            corpus: Training corpus
            model: Pre-created model (creates from config if None)

        Returns:
            Trained model
        """
        # Create model if not provided
        if model is None:
            model = create_model(
                name=self.config.model.name,
                num_topics=self.config.model.num_topics,
                alpha=self.config.model.alpha,
                beta=self.config.model.beta,
                random_state=self.config.seed,
            )

        # Train
        model.fit(
            documents=corpus.documents,
            num_iterations=self.config.training.num_iterations,
            callbacks=self.callbacks.callbacks,
            verbose=self.config.verbose,
        )

        # Save final model
        final_path = self.output_dir / "final_model.pkl"
        model.save(final_path)

        # Save config
        config_path = self.output_dir / "config.yaml"
        self.config.to_yaml(config_path)

        return model

    def evaluate(
        self,
        model: BaseTopicModel,
        corpus: Corpus,
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model.

        Args:
            model: Trained model
            corpus: Evaluation corpus

        Returns:
            Dictionary of evaluation metrics
        """
        results = {
            "model": model.__class__.__name__,
            "num_topics": model.num_topics,
        }

        # Log likelihood
        try:
            results["log_likelihood"] = model.log_likelihood()
        except Exception:
            pass

        # Perplexity
        try:
            results["perplexity"] = model.perplexity()
        except Exception:
            pass

        return results


def train_model(
    corpus: Corpus,
    config: Optional[Config] = None,
    **kwargs: Any,
) -> BaseTopicModel:
    """
    Convenience function to train a model.

    Args:
        corpus: Training corpus
        config: Configuration (creates default if None)
        **kwargs: Override config parameters

    Returns:
        Trained model
    """
    if config is None:
        config = Config()

    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config.model, key):
            setattr(config.model, key, value)
        elif hasattr(config.training, key):
            setattr(config.training, key, value)
        elif hasattr(config, key):
            setattr(config, key, value)

    trainer = Trainer(config)
    return trainer.fit(corpus)
