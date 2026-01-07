"""Benchmark utilities for comparing topic models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import json
import time

import numpy as np

from nbtm.data import Corpus
from nbtm.evaluation.coherence import TopicCoherence
from nbtm.evaluation.perplexity import compute_perplexity
from nbtm.evaluation.diversity import compute_topic_diversity, compute_topic_uniqueness

if TYPE_CHECKING:
    from nbtm.models.base import BaseTopicModel


@dataclass
class BenchmarkResult:
    """Results from a single model benchmark."""

    model_name: str
    num_topics: int

    # Metrics
    coherence: Optional[float] = None
    perplexity: Optional[float] = None
    diversity: Optional[float] = None
    uniqueness: Optional[List[float]] = None
    log_likelihood: Optional[float] = None

    # Timing
    training_time: float = 0.0

    # Per-topic metrics
    topic_coherences: Optional[List[float]] = None

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "num_topics": self.num_topics,
            "coherence": self.coherence,
            "perplexity": self.perplexity,
            "diversity": self.diversity,
            "uniqueness": self.uniqueness,
            "log_likelihood": self.log_likelihood,
            "training_time": self.training_time,
            "topic_coherences": self.topic_coherences,
            "config": self.config,
        }


class BenchmarkRunner:
    """
    Run benchmarks across multiple topic models.

    Provides standardized evaluation and comparison.

    Example:
        >>> runner = BenchmarkRunner(corpus)
        >>> runner.add_model("lda_gibbs", num_topics=10)
        >>> runner.add_model("hdp")
        >>> results = runner.run()
    """

    def __init__(
        self,
        corpus: Corpus,
        coherence_measure: str = "c_v",
        top_n: int = 10,
    ) -> None:
        """
        Initialize benchmark runner.

        Args:
            corpus: Evaluation corpus
            coherence_measure: Coherence metric to use
            top_n: Top words for coherence/diversity
        """
        self.corpus = corpus
        self.coherence_measure = coherence_measure
        self.top_n = top_n

        self.models: List[tuple[str, Dict[str, Any]]] = []
        self.results: List[BenchmarkResult] = []

        # Precompute coherence calculator
        self._coherence = TopicCoherence(
            documents=corpus.documents,
            measure=coherence_measure,
            top_n=top_n,
        )

    def add_model(
        self,
        model_name: str,
        num_iterations: int = 500,
        **kwargs: Any,
    ) -> BenchmarkRunner:
        """
        Add a model configuration to benchmark.

        Args:
            model_name: Model type (e.g., "lda_gibbs", "hdp")
            num_iterations: Training iterations
            **kwargs: Additional model parameters

        Returns:
            self (for chaining)
        """
        config = {
            "model_name": model_name,
            "num_iterations": num_iterations,
            **kwargs,
        }
        self.models.append((model_name, config))
        return self

    def run(
        self,
        verbose: bool = True,
        random_state: int = 42,
    ) -> List[BenchmarkResult]:
        """
        Run benchmark on all configured models.

        Args:
            verbose: Print progress
            random_state: Random seed

        Returns:
            List of benchmark results
        """
        from nbtm.models import create_model

        self.results = []

        for model_name, config in self.models:
            if verbose:
                print(f"\nBenchmarking {model_name}...")

            # Create model
            model_kwargs = {
                k: v for k, v in config.items()
                if k not in ["model_name", "num_iterations"]
            }
            model_kwargs["random_state"] = random_state

            model = create_model(model_name, **model_kwargs)

            # Train
            start_time = time.time()
            model.fit(
                self.corpus.documents,
                num_iterations=config.get("num_iterations", 500),
                verbose=verbose,
            )
            training_time = time.time() - start_time

            # Evaluate
            result = self._evaluate_model(model, training_time, config)
            self.results.append(result)

            if verbose:
                self._print_result(result)

        return self.results

    def _evaluate_model(
        self,
        model: BaseTopicModel,
        training_time: float,
        config: Dict[str, Any],
    ) -> BenchmarkResult:
        """Evaluate a trained model."""
        topic_words = model.get_all_topic_words(self.top_n)

        # Coherence
        coherence = self._coherence.compute(topic_words)
        topic_coherences = self._coherence.compute_per_topic(topic_words)

        # Perplexity
        perplexity = compute_perplexity(model)

        # Diversity
        diversity = compute_topic_diversity(model, self.top_n)
        uniqueness = compute_topic_uniqueness(model, self.top_n)

        # Log-likelihood
        try:
            log_likelihood = model.log_likelihood()
        except Exception:
            log_likelihood = None

        return BenchmarkResult(
            model_name=model.__class__.__name__,
            num_topics=model.num_topics,
            coherence=coherence,
            perplexity=perplexity,
            diversity=diversity,
            uniqueness=uniqueness,
            log_likelihood=log_likelihood,
            training_time=training_time,
            topic_coherences=topic_coherences,
            config=config,
        )

    def _print_result(self, result: BenchmarkResult) -> None:
        """Print benchmark result."""
        print(f"  Topics: {result.num_topics}")
        print(f"  Coherence ({self.coherence_measure}): {result.coherence:.4f}")
        print(f"  Perplexity: {result.perplexity:.2f}")
        print(f"  Diversity: {result.diversity:.4f}")
        print(f"  Training time: {result.training_time:.1f}s")

    def compare(self) -> Dict[str, Any]:
        """
        Compare all benchmark results.

        Returns:
            Comparison summary
        """
        if not self.results:
            return {}

        comparison = {
            "models": [r.model_name for r in self.results],
            "coherence": [r.coherence for r in self.results],
            "perplexity": [r.perplexity for r in self.results],
            "diversity": [r.diversity for r in self.results],
            "training_time": [r.training_time for r in self.results],
        }

        # Find best
        coherence_values = [r for r in comparison["coherence"] if r is not None]
        if coherence_values:
            best_coherence_idx = np.argmax(coherence_values)
            comparison["best_coherence"] = comparison["models"][best_coherence_idx]

        perplexity_values = [r for r in comparison["perplexity"] if r is not None]
        if perplexity_values:
            best_perplexity_idx = np.argmin(perplexity_values)
            comparison["best_perplexity"] = comparison["models"][best_perplexity_idx]

        return comparison

    def save_results(self, path: Path | str) -> None:
        """Save benchmark results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "results": [r.to_dict() for r in self.results],
            "comparison": self.compare(),
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def get_best_model(
        self,
        metric: str = "coherence",
    ) -> Optional[str]:
        """
        Get the best model by metric.

        Args:
            metric: "coherence", "perplexity", or "diversity"

        Returns:
            Name of best model
        """
        if not self.results:
            return None

        if metric == "coherence":
            values = [r.coherence for r in self.results]
            best_idx = np.argmax([v if v is not None else -np.inf for v in values])
        elif metric == "perplexity":
            values = [r.perplexity for r in self.results]
            best_idx = np.argmin([v if v is not None else np.inf for v in values])
        else:  # diversity
            values = [r.diversity for r in self.results]
            best_idx = np.argmax([v if v is not None else -np.inf for v in values])

        return self.results[best_idx].model_name
