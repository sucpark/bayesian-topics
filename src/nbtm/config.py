"""Configuration management using dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import yaml


@dataclass
class ModelConfig:
    """Model configuration."""

    name: Literal["lda_gibbs", "lda_vi", "hdp", "ctm"] = "lda_gibbs"

    # Common parameters
    num_topics: int = 10
    alpha: float = 0.1  # Document-topic Dirichlet prior
    beta: float = 0.01  # Topic-word Dirichlet prior

    # HDP-specific parameters
    gamma: float = 1.0  # Top-level DP concentration
    alpha0: float = 1.0  # Document-level DP concentration

    # CTM-specific parameters
    mean: Optional[list[float]] = None
    covariance: Optional[list[list[float]]] = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.num_topics < 1:
            raise ValueError("num_topics must be at least 1")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if self.beta <= 0:
            raise ValueError("beta must be positive")


@dataclass
class TrainingConfig:
    """Training configuration."""

    num_iterations: int = 1000
    burn_in: int = 200  # Iterations to discard for sampling
    thinning: int = 10  # Keep every N-th sample

    # Convergence
    convergence_threshold: float = 1e-4
    check_convergence_every: int = 100

    # Checkpointing
    save_every: int = 100
    log_every: int = 50

    # Early stopping
    early_stopping: bool = False
    patience: int = 5

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.num_iterations < 1:
            raise ValueError("num_iterations must be at least 1")
        if self.burn_in < 0:
            raise ValueError("burn_in must be non-negative")
        if self.thinning < 1:
            raise ValueError("thinning must be at least 1")


@dataclass
class DataConfig:
    """Data configuration."""

    data_dir: str = "data"
    corpus_file: Optional[str] = None

    # Preprocessing
    min_word_count: int = 5
    max_vocab_size: Optional[int] = None
    stopwords: bool = True
    lowercase: bool = True

    # Train/test split
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_state: int = 42

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.min_word_count < 1:
            raise ValueError("min_word_count must be at least 1")
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"train_ratio + val_ratio + test_ratio must equal 1.0, got {total_ratio}"
            )


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""

    compute_coherence: bool = True
    coherence_measure: Literal["u_mass", "c_v", "c_uci", "c_npmi"] = "c_v"
    top_n_words: int = 10

    compute_perplexity: bool = True
    held_out_ratio: float = 0.1

    compute_diversity: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.top_n_words < 1:
            raise ValueError("top_n_words must be at least 1")
        if not 0 < self.held_out_ratio < 1:
            raise ValueError("held_out_ratio must be between 0 and 1")


@dataclass
class Config:
    """Complete configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # General settings
    seed: int = 42
    output_dir: str = "outputs"
    experiment_name: str = "topic_model_experiment"

    # Logging
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    verbose: bool = True

    @classmethod
    def from_yaml(cls, path: Path | str) -> Config:
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            Config instance
        """
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data or {})

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Config:
        """Create Config from dictionary."""
        model_data = data.pop("model", {})
        training_data = data.pop("training", {})
        data_config = data.pop("data", {})
        evaluation_data = data.pop("evaluation", {})

        return cls(
            model=ModelConfig(**model_data),
            training=TrainingConfig(**training_data),
            data=DataConfig(**data_config),
            evaluation=EvaluationConfig(**evaluation_data),
            **data,
        )

    def to_yaml(self, path: Path | str) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Path to save YAML configuration
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                asdict(self),
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Config(\n"
            f"  model={self.model.name}, num_topics={self.model.num_topics},\n"
            f"  iterations={self.training.num_iterations},\n"
            f"  seed={self.seed}\n"
            f")"
        )
