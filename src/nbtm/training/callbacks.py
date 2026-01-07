"""Training callbacks for topic models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import json
import time

if TYPE_CHECKING:
    from nbtm.models.base import BaseTopicModel


class Callback(ABC):
    """
    Abstract base class for training callbacks.

    Callbacks are used to perform actions at various stages
    of the training process.
    """

    def on_train_begin(self, model: BaseTopicModel) -> None:
        """Called at the beginning of training."""
        pass

    def on_train_end(self, model: BaseTopicModel) -> None:
        """Called at the end of training."""
        pass

    def on_iteration_begin(self, iteration: int, model: BaseTopicModel) -> None:
        """Called at the beginning of each iteration."""
        pass

    def on_iteration_end(self, iteration: int, model: BaseTopicModel) -> None:
        """Called at the end of each iteration."""
        pass


class CallbackList:
    """Container for multiple callbacks."""

    def __init__(self, callbacks: Optional[List[Callback]] = None) -> None:
        """Initialize callback list."""
        self.callbacks = callbacks or []

    def append(self, callback: Callback) -> None:
        """Add a callback."""
        self.callbacks.append(callback)

    def on_train_begin(self, model: BaseTopicModel) -> None:
        """Call on_train_begin for all callbacks."""
        for cb in self.callbacks:
            cb.on_train_begin(model)

    def on_train_end(self, model: BaseTopicModel) -> None:
        """Call on_train_end for all callbacks."""
        for cb in self.callbacks:
            cb.on_train_end(model)

    def on_iteration_begin(self, iteration: int, model: BaseTopicModel) -> None:
        """Call on_iteration_begin for all callbacks."""
        for cb in self.callbacks:
            cb.on_iteration_begin(iteration, model)

    def on_iteration_end(self, iteration: int, model: BaseTopicModel) -> None:
        """Call on_iteration_end for all callbacks."""
        for cb in self.callbacks:
            cb.on_iteration_end(iteration, model)


class EarlyStopping(Callback):
    """
    Stop training when a metric stops improving.

    Attributes:
        patience: Number of iterations to wait before stopping
        min_delta: Minimum change to qualify as improvement
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "max",
    ) -> None:
        """
        Initialize early stopping.

        Args:
            patience: Iterations without improvement before stopping
            min_delta: Minimum improvement threshold
            mode: 'max' or 'min' - whether to maximize or minimize metric
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_value: Optional[float] = None
        self.wait: int = 0
        self.stopped_iteration: int = 0
        self.should_stop: bool = False

    def on_train_begin(self, model: BaseTopicModel) -> None:
        """Reset state at training start."""
        self.best_value = None
        self.wait = 0
        self.should_stop = False

    def on_iteration_end(self, iteration: int, model: BaseTopicModel) -> None:
        """Check if training should stop."""
        try:
            current = model.log_likelihood()
        except Exception:
            return

        if self.best_value is None:
            self.best_value = current
            return

        if self.mode == "max":
            improved = current > self.best_value + self.min_delta
        else:
            improved = current < self.best_value - self.min_delta

        if improved:
            self.best_value = current
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            self.should_stop = True
            self.stopped_iteration = iteration


class ModelCheckpoint(Callback):
    """
    Save model checkpoints during training.

    Attributes:
        filepath: Path to save checkpoints
        save_best_only: Only save when metric improves
        save_every: Save every N iterations
    """

    def __init__(
        self,
        filepath: str | Path,
        save_best_only: bool = True,
        save_every: int = 0,
        mode: str = "max",
    ) -> None:
        """
        Initialize checkpoint callback.

        Args:
            filepath: Save path (can include {iteration} placeholder)
            save_best_only: Only save on improvement
            save_every: Save frequency (0 = only on improvement)
            mode: 'max' or 'min' for metric comparison
        """
        self.filepath = Path(filepath)
        self.save_best_only = save_best_only
        self.save_every = save_every
        self.mode = mode

        self.best_value: Optional[float] = None

    def on_train_begin(self, model: BaseTopicModel) -> None:
        """Reset state and create directory."""
        self.best_value = None
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def on_iteration_end(self, iteration: int, model: BaseTopicModel) -> None:
        """Save checkpoint if appropriate."""
        # Save every N iterations
        if self.save_every > 0 and (iteration + 1) % self.save_every == 0:
            path = str(self.filepath).format(iteration=iteration)
            model.save(path)
            return

        # Save best only
        if self.save_best_only:
            try:
                current = model.log_likelihood()
            except Exception:
                return

            if self.best_value is None:
                self.best_value = current
                model.save(self.filepath)
                return

            if self.mode == "max":
                improved = current > self.best_value
            else:
                improved = current < self.best_value

            if improved:
                self.best_value = current
                model.save(self.filepath)

    def on_train_end(self, model: BaseTopicModel) -> None:
        """Save final model."""
        if not self.save_best_only:
            model.save(self.filepath)


class ProgressLogger(Callback):
    """
    Log training progress.

    Prints metrics at regular intervals.
    """

    def __init__(
        self,
        log_every: int = 50,
        metrics: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize progress logger.

        Args:
            log_every: Logging frequency
            metrics: Metrics to log (default: log_likelihood)
        """
        self.log_every = log_every
        self.metrics = metrics or ["log_likelihood"]
        self._start_time: float = 0

    def on_train_begin(self, model: BaseTopicModel) -> None:
        """Record start time."""
        self._start_time = time.time()
        print(f"Training {model.__class__.__name__}...")

    def on_iteration_end(self, iteration: int, model: BaseTopicModel) -> None:
        """Log metrics."""
        if (iteration + 1) % self.log_every != 0:
            return

        elapsed = time.time() - self._start_time
        msg = f"Iteration {iteration + 1} | Time: {elapsed:.1f}s"

        for metric in self.metrics:
            if metric == "log_likelihood":
                try:
                    value = model.log_likelihood()
                    msg += f" | LL: {value:.2f}"
                except Exception:
                    pass
            elif metric == "num_topics" and hasattr(model, "_active_topics"):
                msg += f" | K: {len(model._active_topics)}"

        print(msg)

    def on_train_end(self, model: BaseTopicModel) -> None:
        """Print final summary."""
        elapsed = time.time() - self._start_time
        print(f"Training complete in {elapsed:.1f}s")


class HistoryLogger(Callback):
    """
    Record training history to file.

    Saves metrics at each iteration to a JSON file.
    """

    def __init__(
        self,
        filepath: str | Path,
        log_every: int = 10,
    ) -> None:
        """
        Initialize history logger.

        Args:
            filepath: Path to save history JSON
            log_every: Logging frequency
        """
        self.filepath = Path(filepath)
        self.log_every = log_every
        self.history: List[Dict[str, Any]] = []

    def on_train_begin(self, model: BaseTopicModel) -> None:
        """Reset history."""
        self.history = []
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def on_iteration_end(self, iteration: int, model: BaseTopicModel) -> None:
        """Record metrics."""
        if (iteration + 1) % self.log_every != 0:
            return

        record = {
            "iteration": iteration + 1,
            "timestamp": time.time(),
        }

        try:
            record["log_likelihood"] = model.log_likelihood()
        except Exception:
            pass

        if hasattr(model, "_active_topics"):
            record["num_topics"] = len(model._active_topics)

        self.history.append(record)

    def on_train_end(self, model: BaseTopicModel) -> None:
        """Save history to file."""
        with open(self.filepath, "w") as f:
            json.dump(self.history, f, indent=2)


class WandbLogger(Callback):
    """
    Log metrics to Weights & Biases.

    Requires wandb to be installed and configured.
    """

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        log_every: int = 10,
    ) -> None:
        """
        Initialize W&B logger.

        Args:
            project: W&B project name
            name: Run name
            config: Configuration to log
            log_every: Logging frequency
        """
        self.project = project
        self.name = name
        self.config = config or {}
        self.log_every = log_every
        self._run = None

    def on_train_begin(self, model: BaseTopicModel) -> None:
        """Initialize W&B run."""
        try:
            import wandb

            config = {**model.get_config(), **self.config}
            self._run = wandb.init(
                project=self.project,
                name=self.name,
                config=config,
            )
        except ImportError:
            print("wandb not installed. Skipping W&B logging.")

    def on_iteration_end(self, iteration: int, model: BaseTopicModel) -> None:
        """Log metrics to W&B."""
        if self._run is None:
            return

        if (iteration + 1) % self.log_every != 0:
            return

        import wandb

        metrics = {"iteration": iteration + 1}

        try:
            metrics["log_likelihood"] = model.log_likelihood()
        except Exception:
            pass

        if hasattr(model, "_active_topics"):
            metrics["num_topics"] = len(model._active_topics)

        wandb.log(metrics)

    def on_train_end(self, model: BaseTopicModel) -> None:
        """Finish W&B run."""
        if self._run is not None:
            import wandb

            wandb.finish()
