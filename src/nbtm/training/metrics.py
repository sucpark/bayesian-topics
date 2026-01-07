"""Training metrics for topic models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TrainingMetrics:
    """
    Container for training metrics.

    Tracks metrics across training iterations.
    """

    iterations: List[int] = field(default_factory=list)
    log_likelihood: List[float] = field(default_factory=list)
    perplexity: List[float] = field(default_factory=list)
    num_topics: List[int] = field(default_factory=list)
    elapsed_time: List[float] = field(default_factory=list)

    def add(
        self,
        iteration: int,
        log_likelihood: Optional[float] = None,
        perplexity: Optional[float] = None,
        num_topics: Optional[int] = None,
        elapsed_time: Optional[float] = None,
    ) -> None:
        """Add metrics for an iteration."""
        self.iterations.append(iteration)

        if log_likelihood is not None:
            self.log_likelihood.append(log_likelihood)
        if perplexity is not None:
            self.perplexity.append(perplexity)
        if num_topics is not None:
            self.num_topics.append(num_topics)
        if elapsed_time is not None:
            self.elapsed_time.append(elapsed_time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "iterations": self.iterations,
            "log_likelihood": self.log_likelihood,
            "perplexity": self.perplexity,
            "num_topics": self.num_topics,
            "elapsed_time": self.elapsed_time,
        }

    @property
    def best_log_likelihood(self) -> Optional[float]:
        """Get best (highest) log-likelihood."""
        if not self.log_likelihood:
            return None
        return max(self.log_likelihood)

    @property
    def best_perplexity(self) -> Optional[float]:
        """Get best (lowest) perplexity."""
        if not self.perplexity:
            return None
        return min(self.perplexity)

    @property
    def final_num_topics(self) -> Optional[int]:
        """Get final number of topics."""
        if not self.num_topics:
            return None
        return self.num_topics[-1]

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        summary = {
            "total_iterations": len(self.iterations),
        }

        if self.log_likelihood:
            summary["best_log_likelihood"] = self.best_log_likelihood
            summary["final_log_likelihood"] = self.log_likelihood[-1]

        if self.perplexity:
            summary["best_perplexity"] = self.best_perplexity
            summary["final_perplexity"] = self.perplexity[-1]

        if self.num_topics:
            summary["final_num_topics"] = self.final_num_topics

        if self.elapsed_time:
            summary["total_time"] = sum(self.elapsed_time)

        return summary


class MetricsTracker:
    """
    Track and aggregate metrics across experiments.

    Useful for comparing multiple models or hyperparameter settings.
    """

    def __init__(self) -> None:
        """Initialize tracker."""
        self.experiments: Dict[str, TrainingMetrics] = {}

    def add_experiment(self, name: str) -> TrainingMetrics:
        """Add a new experiment."""
        self.experiments[name] = TrainingMetrics()
        return self.experiments[name]

    def get_experiment(self, name: str) -> Optional[TrainingMetrics]:
        """Get metrics for an experiment."""
        return self.experiments.get(name)

    def compare(self) -> Dict[str, Dict[str, Any]]:
        """Compare all experiments."""
        return {
            name: metrics.summary()
            for name, metrics in self.experiments.items()
        }

    def best_experiment(self, metric: str = "log_likelihood") -> Optional[str]:
        """
        Get the best experiment by metric.

        Args:
            metric: Metric to compare ('log_likelihood' or 'perplexity')

        Returns:
            Name of best experiment
        """
        if not self.experiments:
            return None

        best_name = None
        best_value = None

        for name, metrics in self.experiments.items():
            if metric == "log_likelihood":
                value = metrics.best_log_likelihood
                better = value is not None and (best_value is None or value > best_value)
            else:  # perplexity
                value = metrics.best_perplexity
                better = value is not None and (best_value is None or value < best_value)

            if better:
                best_value = value
                best_name = name

        return best_name
