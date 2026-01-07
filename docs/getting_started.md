# Getting Started with NBTM

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nbtm.git
cd nbtm

# Install with pip
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

### Full Installation (all optional dependencies)

```bash
pip install -e ".[all]"
```

## Quick Start

### Python API

```python
from nbtm.models import create_model
from nbtm.data import Corpus

# Prepare your documents
documents = [
    ["machine", "learning", "algorithms", "data"],
    ["deep", "learning", "neural", "networks"],
    ["statistics", "probability", "inference"],
    # ... more documents
]

# Create corpus
corpus = Corpus(documents=documents)

# Create and train model
model = create_model("lda_gibbs", num_topics=10)
model.fit(corpus.documents, num_iterations=500)

# Get results
for k in range(model.num_topics):
    words = model.get_topic_words(k, top_n=5)
    print(f"Topic {k}: {[w for w, _ in words]}")
```

### Command Line

```bash
# Train a model
nbtm train --config configs/default.yaml

# Evaluate model
nbtm evaluate --model-path outputs/final_model.pkl

# Generate visualizations
nbtm visualize --model-path outputs/final_model.pkl
```

## Available Models

| Model | Command | Description |
|-------|---------|-------------|
| Gibbs LDA | `lda_gibbs` | Collapsed Gibbs Sampling |
| Variational LDA | `lda_vi` | Mean-field variational inference |
| HDP | `hdp` | Hierarchical Dirichlet Process |
| CTM | `ctm` | Correlated Topic Model |

## Configuration

Create a configuration file:

```yaml
model:
  name: lda_gibbs
  num_topics: 10
  alpha: 0.1
  beta: 0.01

training:
  num_iterations: 1000
  log_every: 100

seed: 42
output_dir: outputs
```

## Next Steps

- Read [Algorithms](algorithms.md) for detailed algorithm descriptions
- See [Configuration](configuration.md) for all options
- Check [Evaluation](evaluation.md) for metrics
