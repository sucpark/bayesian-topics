# NBTM: Nonparametric Bayesian Topic Modeling

A research framework for comparing topic modeling algorithms, including LDA variants and nonparametric Bayesian models.

## Features

- **Multiple Algorithms**: Gibbs Sampling LDA, Variational LDA, HDP, CTM
- **Unified Interface**: Common API for all topic models
- **Evaluation Metrics**: Topic coherence (UMass, C_V, NPMI), perplexity, diversity
- **Visualization**: Word clouds, topic-word distributions, convergence plots
- **CLI Support**: Easy experiment management from command line
- **Experiment Tracking**: Optional WandB/MLflow integration

## Installation

```bash
# Basic installation
pip install -e .

# With development dependencies
pip install -e ".[dev]"

# With all optional dependencies
pip install -e ".[all]"
```

## Quick Start

### Python API

```python
from nbtm.models import create_model
from nbtm.data import load_corpus

# Load data
documents = load_corpus("data/raw/corpus.txt")

# Create and train model
model = create_model("lda_gibbs", num_topics=10)
model.fit(documents, num_iterations=1000)

# Get results
topics = model.get_all_topic_words(top_n=10)
for i, topic in enumerate(topics):
    print(f"Topic {i}: {[word for word, _ in topic]}")
```

### Command Line

```bash
# Train a model
nbtm train --config configs/default.yaml --num-topics 10

# Evaluate model
nbtm evaluate --model-path outputs/model.pkl --metrics all

# Generate visualizations
nbtm visualize --model-path outputs/model.pkl --type wordcloud

# List available models
nbtm list-models
```

## Supported Algorithms

| Algorithm | Description | Key Features |
|-----------|-------------|--------------|
| **Gibbs LDA** | Collapsed Gibbs Sampling | Simple, interpretable |
| **LDA-VI** | Variational Inference | Fast, scalable |
| **HDP** | Hierarchical Dirichlet Process | Automatic topic count |
| **CTM** | Correlated Topic Model | Topic correlations |

## Project Structure

```
nbtm/
├── src/nbtm/           # Main package
│   ├── models/         # Topic model implementations
│   ├── data/           # Data loading and preprocessing
│   ├── training/       # Training infrastructure
│   ├── evaluation/     # Evaluation metrics
│   ├── visualization/  # Plotting tools
│   └── utils/          # Utilities
├── configs/            # YAML configurations
├── notebooks/          # Tutorial notebooks
├── docs/               # Documentation
└── tests/              # Test suite
```

## Configuration

Example configuration (`configs/default.yaml`):

```yaml
model:
  name: lda_gibbs
  num_topics: 10
  alpha: 0.1
  beta: 0.01

training:
  num_iterations: 1000
  burn_in: 200

evaluation:
  compute_coherence: true
  coherence_measure: c_v
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/
ruff check src/ --fix

# Type check
mypy src/
```

## License

MIT License

## References

- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation. JMLR.
- Teh, Y. W., et al. (2006). Hierarchical Dirichlet Processes. JASA.
- Blei, D. M., & Lafferty, J. D. (2007). Correlated Topic Models. NIPS.
