# Topic Modeling Algorithms

## Latent Dirichlet Allocation (LDA)

LDA assumes each document is a mixture of topics, and each topic is a distribution over words.

### Gibbs Sampling LDA

**Command:** `lda_gibbs`

Uses collapsed Gibbs sampling where topic-word and document-topic distributions are integrated out.

**Parameters:**
- `num_topics` (K): Number of topics
- `alpha`: Document-topic Dirichlet prior (default: 0.1)
- `beta`: Topic-word Dirichlet prior (default: 0.01)

**Pros:**
- Simple to understand and implement
- Produces samples from true posterior
- Good for small to medium datasets

**Cons:**
- Slow for large datasets
- Stochastic (different runs may give different results)

### Variational LDA

**Command:** `lda_vi`

Uses mean-field variational inference to approximate the posterior.

**Parameters:**
- Same as Gibbs LDA

**Pros:**
- Faster than Gibbs sampling
- Deterministic optimization
- Scales to large datasets

**Cons:**
- Approximation to true posterior
- May get stuck in local optima

## Hierarchical Dirichlet Process (HDP)

**Command:** `hdp`

A nonparametric extension of LDA that automatically infers the number of topics.

**Parameters:**
- `gamma`: Top-level DP concentration (default: 1.0)
- `alpha0`: Document-level DP concentration (default: 1.0)
- `beta`: Topic-word prior (default: 0.01)

**Pros:**
- No need to specify number of topics
- Can discover structure in data
- Theoretically principled

**Cons:**
- More complex inference
- Slower than fixed-topic models
- Sensitive to concentration parameters

## Correlated Topic Model (CTM)

**Command:** `ctm`

Uses a logistic normal distribution instead of Dirichlet, allowing topics to be correlated.

**Parameters:**
- `num_topics`: Number of topics
- `beta`: Topic-word prior

**Pros:**
- Models topic correlations
- More realistic for many domains
- Provides correlation matrix

**Cons:**
- More parameters to estimate
- Slower inference
- No conjugacy

## Choosing a Model

| Scenario | Recommended Model |
|----------|-------------------|
| First experiment | `lda_gibbs` |
| Large dataset | `lda_vi` |
| Unknown topic count | `hdp` |
| Topic relationships important | `ctm` |
| Speed is critical | `lda_vi` |
| Need interpretable samples | `lda_gibbs` |
