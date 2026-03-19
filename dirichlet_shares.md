# Using Dirichlet Distribution for Anti-Correlated Shares

## The Problem

When modeling chip production shares within a quarter (e.g., v5e: 85-95%, v5p: 5-15%), sampling each share independently and normalizing doesn't capture the anti-correlation: if v5e is high, v5p *should* be low.

Current approach:
```python
raw_shares = {version: dist @ 1 for version, dist in mix.items()}
return normalize_shares(raw_shares)  # independent samples, then normalize
```

## Dirichlet Distribution

The Dirichlet is the multivariate generalization of the Beta distribution. It produces vectors of K values that **always sum to 1** - perfect for shares.

### Parameters

Parameterized by concentration vector **α = (α₁, α₂, ..., αₖ)**:

- **Mean** for component i: `αᵢ / sum(α)`
- **Concentration** (sum of all α): higher = tighter around mean, lower = more variance
- Components are **naturally anti-correlated**: if one is high, others must be lower

### Basic Example

```python
import numpy as np

# Example: 2 chips, want ~90% v5e, ~10% v5p
# Mean = α / sum(α), so α = [9, 1] gives means [0.9, 0.1]

alpha = [9, 1]
samples = np.random.dirichlet(alpha, size=5000)
# samples.shape = (5000, 2), each row sums to 1

v5e_shares = samples[:, 0]  # mean ≈ 0.9
v5p_shares = samples[:, 1]  # mean ≈ 0.1

print(f"v5e: {v5e_shares.mean():.3f}, v5p: {v5p_shares.mean():.3f}")
print(f"Correlation: {np.corrcoef(v5e_shares, v5p_shares)[0,1]:.3f}")  # -1.0 for K=2
```

### Controlling Variance

The **sum of alphas** controls how tight the distribution is:

```python
# Same 90/10 mean, different concentrations:
tight = np.random.dirichlet([90, 10], size=5000)    # sum=100, tight
loose = np.random.dirichlet([0.9, 0.1], size=5000)  # sum=1, very spread out
medium = np.random.dirichlet([9, 1], size=5000)     # sum=10, moderate

print(f"Tight std:  {tight[:,0].std():.3f}")   # ~0.03
print(f"Medium std: {medium[:,0].std():.3f}")  # ~0.09
print(f"Loose std:  {loose[:,0].std():.3f}")   # ~0.27
```

## Potential Implementation for TPU Shares

```python
def sample_shares_dirichlet(quarter):
    """Sample shares using Dirichlet for proper anti-correlation."""
    mix = PROD_MIX[quarter]
    versions = list(mix.keys())

    # Convert p5/p95 ranges to Dirichlet alphas
    # Use midpoint as mean, scale for desired variance
    means = []
    for v in versions:
        dist = mix[v]
        midpoint = (dist.x + dist.y) / 2  # rough mean from the range
        means.append(midpoint)

    # Normalize means to sum to 1
    total = sum(means)
    means = [m / total for m in means]

    # Concentration controls variance - higher = tighter
    # Adjust based on how wide your p5-p95 ranges are
    concentration = 20  # tune this

    alphas = [m * concentration for m in means]

    # Sample once
    shares = np.random.dirichlet(alphas)

    return {v: shares[i] for i, v in enumerate(versions)}
```

## Calibration Challenge

The tricky part is **mapping existing p5/p95 ranges to Dirichlet parameters**. Options:

1. **Empirical tuning**: Sample many times, check if marginal percentiles roughly match your intended ranges
2. **Moment matching**: Use formulas for Dirichlet variance to solve for concentration
3. **Accept approximation**: Dirichlet may not perfectly match arbitrary marginal ranges, but captures the key property (anti-correlation)

## When It Matters

Anti-correlation matters most when:
- Computing **totals across chip types** within a quarter (errors don't cancel)
- Chip types have **similar uncertainty ranges** (independent sampling inflates total variance)
- You care about **joint distribution** not just marginals
