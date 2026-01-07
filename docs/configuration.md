# Configuration Guide

## Configuration File Structure

```yaml
# Model settings
model:
  name: lda_gibbs        # Algorithm: lda_gibbs, lda_vi, hdp, ctm
  num_topics: 10         # Number of topics (K)
  alpha: 0.1             # Document-topic prior
  beta: 0.01             # Topic-word prior

  # HDP-specific
  gamma: 1.0             # Top-level DP concentration
  alpha0: 1.0            # Document-level DP concentration

# Training settings
training:
  num_iterations: 1000   # Number of iterations
  burn_in: 200           # Iterations to discard (Gibbs)
  thinning: 10           # Keep every N-th sample

  convergence_threshold: 0.0001
  check_convergence_every: 100

  save_every: 100        # Checkpoint frequency
  log_every: 50          # Logging frequency

  early_stopping: false
  patience: 5

# Data settings
data:
  data_dir: data
  corpus_file: null      # Path relative to data_dir

  min_word_count: 5      # Minimum word frequency
  max_vocab_size: null   # Maximum vocabulary size
  stopwords: true        # Remove stopwords
  lowercase: true        # Convert to lowercase

  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  random_state: 42

# Evaluation settings
evaluation:
  compute_coherence: true
  coherence_measure: c_v  # u_mass, c_v, c_uci, c_npmi
  top_n_words: 10

  compute_perplexity: true
  held_out_ratio: 0.1

  compute_diversity: true

# General settings
seed: 42
output_dir: outputs
experiment_name: my_experiment
use_wandb: false
verbose: true
```

## Model-Specific Configurations

### Gibbs LDA

```yaml
model:
  name: lda_gibbs
  num_topics: 10
  alpha: 0.1
  beta: 0.01

training:
  num_iterations: 1000
  burn_in: 200
  thinning: 10
```

### Variational LDA

```yaml
model:
  name: lda_vi
  num_topics: 10
  alpha: 0.1
  beta: 0.01

training:
  num_iterations: 100
  convergence_threshold: 0.001
```

### HDP

```yaml
model:
  name: hdp
  num_topics: 50  # Maximum/initial topics
  gamma: 1.0
  alpha0: 1.0
  beta: 0.01

training:
  num_iterations: 500
```

### CTM

```yaml
model:
  name: ctm
  num_topics: 10
  beta: 0.01

training:
  num_iterations: 200
```

## CLI Override

Override configuration from command line:

```bash
nbtm train --config configs/default.yaml \
    --num-topics 20 \
    --iterations 500 \
    --model lda_vi \
    --seed 123
```

## Programmatic Configuration

```python
from nbtm.config import Config, ModelConfig, TrainingConfig

config = Config(
    model=ModelConfig(name="lda_gibbs", num_topics=15),
    training=TrainingConfig(num_iterations=500),
    seed=42,
)

# Save to file
config.to_yaml("my_config.yaml")

# Load from file
config = Config.from_yaml("my_config.yaml")
```
