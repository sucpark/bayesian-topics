"""Command-line interface for NBTM."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from nbtm import __version__


@click.group()
@click.version_option(version=__version__)
def main():
    """
    NBTM: Nonparametric Bayesian Topic Modeling

    A research framework for comparing topic modeling algorithms.

    Examples:

        # Train a model
        nbtm train --config configs/default.yaml

        # Evaluate a trained model
        nbtm evaluate --model-path outputs/model.pkl

        # Generate visualizations
        nbtm visualize --model-path outputs/model.pkl

        # List available models
        nbtm list-models
    """
    pass


@main.command()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    required=True,
    help="Path to configuration YAML file"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default="outputs",
    help="Output directory"
)
@click.option(
    "--num-topics", "-k",
    type=int,
    default=None,
    help="Override number of topics"
)
@click.option(
    "--iterations", "-n",
    type=int,
    default=None,
    help="Override number of iterations"
)
@click.option(
    "--model", "-m",
    type=click.Choice(["lda_gibbs", "lda_vi", "hdp", "ctm"]),
    default=None,
    help="Override model type"
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed"
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    help="Suppress progress output"
)
def train(
    config: str,
    output: str,
    num_topics: Optional[int],
    iterations: Optional[int],
    model: Optional[str],
    seed: Optional[int],
    quiet: bool,
):
    """Train a topic model."""
    from nbtm.config import Config
    from nbtm.data import Corpus
    from nbtm.training import Trainer

    # Load configuration
    click.echo(f"Loading configuration from {config}...")
    cfg = Config.from_yaml(config)

    # Apply overrides
    if num_topics is not None:
        cfg.model.num_topics = num_topics
    if iterations is not None:
        cfg.training.num_iterations = iterations
    if model is not None:
        cfg.model.name = model
    if seed is not None:
        cfg.seed = seed
    cfg.output_dir = output
    cfg.verbose = not quiet

    # Load data
    if cfg.data.corpus_file:
        corpus_path = Path(cfg.data.data_dir) / cfg.data.corpus_file
        if corpus_path.exists():
            click.echo(f"Loading corpus from {corpus_path}...")
            corpus = Corpus.from_file(
                corpus_path,
                min_word_count=cfg.data.min_word_count,
                max_vocab_size=cfg.data.max_vocab_size,
            )
        else:
            click.echo(f"Corpus file not found: {corpus_path}")
            click.echo("Using sample data...")
            corpus = _get_sample_corpus()
    else:
        click.echo("No corpus file specified. Using sample data...")
        corpus = _get_sample_corpus()

    click.echo(f"Corpus: {len(corpus)} documents, {len(corpus.vocabulary)} vocabulary")
    click.echo(f"Training {cfg.model.name} with {cfg.model.num_topics} topics...")

    # Train
    trainer = Trainer(cfg)
    trained_model = trainer.fit(corpus)

    click.echo(f"\nTraining complete!")
    click.echo(f"Model saved to: {output}/final_model.pkl")

    # Print top words for each topic
    if not quiet:
        click.echo("\nTop words per topic:")
        for k in range(min(5, trained_model.num_topics)):
            words = trained_model.get_topic_words(k, 5)
            word_str = ", ".join(w for w, _ in words)
            click.echo(f"  Topic {k}: {word_str}")

        if trained_model.num_topics > 5:
            click.echo(f"  ... and {trained_model.num_topics - 5} more topics")


@main.command()
@click.option(
    "--model-path", "-m",
    type=click.Path(exists=True),
    required=True,
    help="Path to trained model"
)
@click.option(
    "--data-path", "-d",
    type=click.Path(exists=True),
    default=None,
    help="Path to evaluation data (uses training data if not specified)"
)
@click.option(
    "--metrics",
    type=click.Choice(["coherence", "perplexity", "diversity", "all"]),
    default="all",
    help="Metrics to compute"
)
@click.option(
    "--top-n",
    type=int,
    default=10,
    help="Number of top words for coherence"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="Output file for results (JSON)"
)
def evaluate(
    model_path: str,
    data_path: Optional[str],
    metrics: str,
    top_n: int,
    output: Optional[str],
):
    """Evaluate a trained topic model."""
    import json
    from nbtm.models.base import BaseTopicModel
    from nbtm.evaluation import (
        compute_coherence,
        compute_perplexity,
        compute_topic_diversity,
    )

    click.echo(f"Loading model from {model_path}...")

    # Load model
    model = BaseTopicModel.load(model_path)
    click.echo(f"Model: {model.__class__.__name__}, {model.num_topics} topics")

    # Get documents
    if data_path:
        from nbtm.data import Corpus
        corpus = Corpus.from_file(data_path)
        documents = corpus.documents
    else:
        documents = model._documents

    results = {"model": model.__class__.__name__, "num_topics": model.num_topics}

    # Compute metrics
    if metrics in ["coherence", "all"]:
        click.echo("Computing coherence...")
        coh = compute_coherence(model, documents, measure="c_v", top_n=top_n)
        results["coherence_cv"] = coh
        click.echo(f"  Coherence (C_V): {coh:.4f}")

    if metrics in ["perplexity", "all"]:
        click.echo("Computing perplexity...")
        perp = compute_perplexity(model)
        results["perplexity"] = perp
        click.echo(f"  Perplexity: {perp:.2f}")

    if metrics in ["diversity", "all"]:
        click.echo("Computing diversity...")
        div = compute_topic_diversity(model, top_n=top_n)
        results["diversity"] = div
        click.echo(f"  Diversity: {div:.4f}")

    # Save results
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        click.echo(f"\nResults saved to: {output}")


@main.command()
@click.option(
    "--model-path", "-m",
    type=click.Path(exists=True),
    required=True,
    help="Path to trained model"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default="outputs/visualizations",
    help="Output directory for visualizations"
)
@click.option(
    "--type", "-t", "viz_type",
    type=click.Choice(["topics", "heatmap", "wordcloud", "documents", "all"]),
    default="all",
    help="Type of visualization"
)
@click.option(
    "--top-n",
    type=int,
    default=10,
    help="Number of top words to show"
)
@click.option(
    "--num-topics",
    type=int,
    default=None,
    help="Number of topics to visualize"
)
def visualize(
    model_path: str,
    output: str,
    viz_type: str,
    top_n: int,
    num_topics: Optional[int],
):
    """Generate visualizations for a trained model."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend

    from nbtm.models.base import BaseTopicModel
    from nbtm.visualization import (
        plot_topic_words,
        plot_topic_heatmap,
        plot_document_topics,
    )

    click.echo(f"Loading model from {model_path}...")
    model = BaseTopicModel.load(model_path)

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if viz_type in ["topics", "all"]:
        click.echo("Generating topic-word bar charts...")
        fig = plot_topic_words(model, num_topics=num_topics, top_n=top_n)
        fig.savefig(output_dir / "topic_words.png", dpi=150, bbox_inches='tight')
        click.echo(f"  Saved: topic_words.png")

    if viz_type in ["heatmap", "all"]:
        click.echo("Generating topic-word heatmap...")
        fig = plot_topic_heatmap(model, top_n=top_n)
        fig.savefig(output_dir / "topic_heatmap.png", dpi=150, bbox_inches='tight')
        click.echo(f"  Saved: topic_heatmap.png")

    if viz_type in ["documents", "all"]:
        click.echo("Generating document-topic heatmap...")
        fig = plot_document_topics(model, num_docs=50)
        fig.savefig(output_dir / "document_topics.png", dpi=150, bbox_inches='tight')
        click.echo(f"  Saved: document_topics.png")

    if viz_type in ["wordcloud", "all"]:
        try:
            from nbtm.visualization import plot_all_topic_wordclouds
            click.echo("Generating word clouds...")
            fig = plot_all_topic_wordclouds(model, num_topics=num_topics, top_n=top_n)
            fig.savefig(output_dir / "wordclouds.png", dpi=150, bbox_inches='tight')
            click.echo(f"  Saved: wordclouds.png")
        except ImportError:
            click.echo("  Skipping word clouds (wordcloud package not installed)")

    click.echo(f"\nVisualizations saved to: {output_dir}")


@main.command("list-models")
def list_models():
    """List available topic models."""
    from nbtm.models import get_available_models

    models = get_available_models()

    click.echo("Available topic models:\n")

    descriptions = {
        "lda_gibbs": "Gibbs Sampling LDA - Collapsed Gibbs sampling for LDA",
        "lda_vi": "Variational LDA - Mean-field variational inference",
        "hdp": "HDP - Hierarchical Dirichlet Process (nonparametric)",
        "ctm": "CTM - Correlated Topic Model with logistic normal",
    }

    for name in models:
        desc = descriptions.get(name, "")
        click.echo(f"  {name}")
        if desc:
            click.echo(f"    {desc}")

    click.echo(f"\nTotal: {len(models)} models")


@main.command()
@click.option(
    "--output", "-o",
    type=click.Path(),
    default="configs/default.yaml",
    help="Output path for config file"
)
def init_config(output: str):
    """Create a default configuration file."""
    from nbtm.config import Config

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = Config()
    config.to_yaml(output_path)

    click.echo(f"Created default configuration at: {output_path}")


def _get_sample_corpus():
    """Get sample corpus for testing."""
    from nbtm.data import Corpus

    documents = [
        ["Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"],
        ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"],
        ["Python", "scikit-learn", "scipy", "numpy", "statsmodels", "pandas"],
        ["R", "Python", "statistics", "regression", "probability"],
        ["machine learning", "regression", "decision trees", "libsvm"],
        ["Python", "R", "Java", "C++", "Haskell", "programming languages"],
        ["statistics", "probability", "mathematics", "theory"],
        ["machine learning", "scikit-learn", "Mahout", "neural networks"],
        ["neural networks", "deep learning", "Big Data", "artificial intelligence"],
        ["Hadoop", "Java", "MapReduce", "Big Data"],
        ["statistics", "R", "statsmodels"],
        ["C++", "deep learning", "artificial intelligence", "probability"],
        ["pandas", "R", "Python"],
        ["databases", "HBase", "Postgres", "MySQL", "MongoDB"],
        ["libsvm", "regression", "support vector machines"],
    ]

    return Corpus(documents=documents)


if __name__ == "__main__":
    main()
