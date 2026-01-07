# Evaluation Guide

## Evaluation Metrics

### Topic Coherence

Coherence measures how semantically similar the top words of a topic are.

**Available Measures:**

| Measure | Range | Description |
|---------|-------|-------------|
| `u_mass` | (-∞, 0] | Document co-occurrence based |
| `c_uci` | (-∞, +∞) | Pointwise mutual information |
| `c_npmi` | [-1, 1] | Normalized PMI |
| `c_v` | [0, 1] | Combined measure (recommended) |

**Usage:**

```python
from nbtm.evaluation import compute_coherence

coherence = compute_coherence(
    model,
    documents,
    measure="c_v",
    top_n=10
)
print(f"Coherence: {coherence:.4f}")
```

### Perplexity

Perplexity measures how well the model predicts held-out words. Lower is better.

```python
from nbtm.evaluation import compute_perplexity

perplexity = compute_perplexity(model)
print(f"Perplexity: {perplexity:.2f}")
```

### Topic Diversity

Diversity measures how distinct topics are from each other.

```python
from nbtm.evaluation import (
    compute_topic_diversity,
    compute_topic_uniqueness,
    compute_topic_overlap
)

# Overall diversity (0-1, higher is better)
diversity = compute_topic_diversity(model, top_n=25)

# Per-topic uniqueness
uniqueness = compute_topic_uniqueness(model, top_n=10)

# Pairwise overlap matrix
overlap = compute_topic_overlap(model, top_n=10)
```

## Benchmark Runner

Compare multiple models:

```python
from nbtm.evaluation import BenchmarkRunner
from nbtm.data import Corpus

corpus = Corpus.from_file("data/corpus.txt")

runner = BenchmarkRunner(corpus, coherence_measure="c_v")

# Add models to compare
runner.add_model("lda_gibbs", num_topics=10, num_iterations=500)
runner.add_model("lda_vi", num_topics=10, num_iterations=100)
runner.add_model("hdp", num_iterations=500)

# Run benchmark
results = runner.run()

# Compare results
comparison = runner.compare()
print(comparison)

# Find best model
best = runner.get_best_model(metric="coherence")
print(f"Best model: {best}")

# Save results
runner.save_results("benchmark_results.json")
```

## Evaluation from CLI

```bash
# Evaluate all metrics
nbtm evaluate --model-path outputs/model.pkl --metrics all

# Specific metrics
nbtm evaluate --model-path outputs/model.pkl --metrics coherence

# Save to file
nbtm evaluate --model-path outputs/model.pkl --output results.json
```

## Interpretation Guidelines

### Coherence

- **C_V > 0.5**: Good coherence
- **C_V > 0.6**: Very good coherence
- **C_V > 0.7**: Excellent coherence

### Perplexity

- Lower is better
- Compare models trained on same data
- Typical range: 100-2000 (depends on vocabulary)

### Diversity

- **Diversity > 0.8**: High diversity (topics are distinct)
- **Diversity < 0.5**: Low diversity (topics overlap significantly)
