# Correlated Sampling Across Time Periods (Squigglepy)

## Problem

A parameter varies over time but uncertainty should be correlated. E.g., a company's share of Nvidia is 10-15% one year and 15-20% the next — if the true share is high in year 1, it should tend to be high in year 2.

## Approaches

### 1. `sq.correlate` with a correlation matrix (tunable correlation)

```python
y1 = sq.to(0.10, 0.15)
y2 = sq.to(0.12, 0.17)
y3 = sq.to(0.15, 0.20)

# AR(1)-like structure: correlation decays with distance
rho = 0.7
y1, y2, y3 = sq.correlate((y1, y2, y3), [
    [1.0,      rho,      rho**2],
    [rho,      1.0,      rho   ],
    [rho**2,   rho,      1.0   ],
])

# Sample — high draws tend to stay high, but with drift
samples = [(y1 @ 1, y2 @ 1, y3 @ 1) for _ in range(1000)]
```

- `rho = 0.9`: sticky — high in year 1 means very likely high in year 2
- `rho = 0.5`: moderate independent variation each period
- `rho = 0.0`: fully independent (no correlation)
- `rho**|i-j|` for non-adjacent periods gives natural decay

### 2. Sample once + deterministic deflation (100% correlation)

Used in `chip_estimates_utils.py` for prices. Sample from the base period once, then scale deterministically:

```python
base_share = sq.to(0.10, 0.15) @ 1
year1 = base_share
year2 = base_share * np.sqrt(0.175 / 0.125)  # scale by ratio of geometric means
```

If you're in a "high share world," you stay there. This is what `sample_base_price` + `get_deflation_factor` does throughout the codebase.

### 3. Sample once, reuse across quarters (100% correlation, same value)

Used in `nvidia_owners.ipynb` for hyperscaler shares — sample once per MC iteration, reuse for every quarter:

```python
sampled_shares = {company: dist @ 1 for company, dist in hyperscaler_shares.items()}
for quarter in quarters:
    share = sampled_shares[company]  # same value every quarter
```

## How `sq.correlate` Works Internally

`sq.correlate()` does **not** sample anything upfront. It uses lazy/on-demand sampling:

1. **`sq.correlate()`** creates a `CorrelationGroup` and attaches it to each distribution.

2. **First sample** of any distribution in the group (e.g., `y1 @ 1000`) triggers `sample_correlated_group()`, which:
   - Independently samples all distributions in the group
   - Re-shuffles samples using the **Iman-Conover method** (Cholesky decomposition) to match the target rank-correlation matrix
   - Stores correlated samples for the other distributions in `dist._correlated_samples`

3. **Subsequent samples** of other distributions in the group (e.g., `y2 @ 1000`) return the pre-computed correlated samples and clear the cache.

4. **Next round** of sampling triggers a fresh correlated draw.

**Key constraint**: sample each distribution in the group exactly once per "round" with the same `n`. The pattern is:

```python
y1, y2 = sq.correlate((sq.to(0.10, 0.15), sq.to(0.15, 0.20)), 0.8)

# Batch mode:
a = y1 @ 1000   # triggers sampling of BOTH
b = y2 @ 1000   # returns pre-computed correlated samples

# Per-iteration mode:
for _ in range(1000):
    s1 = y1 @ 1  # triggers correlated sampling
    s2 = y2 @ 1  # returns correlated value
```

## Integration with `estimate_cumulative_chip_sales`

To use tunable correlation in the existing simulation loop, pre-sample correlated values outside the quarter loop, then index into them:

```python
share_dists = [sq.to(lo, hi) for lo, hi in shares_by_quarter.values()]
share_dists = sq.correlate(share_dists, build_ar1_matrix(rho, len(share_dists)))

# In the MC loop:
sampled_shares = [d @ 1 for d in share_dists]
for i, quarter in enumerate(quarters):
    share = sampled_shares[i]
```

## Summary Table

| Method | Correlation | Flexibility |
|--------|------------|-------------|
| `sq.correlate` | Tunable (0 to 1) | Can set exact correlation coefficient |
| Sample once + deflation | Perfect (1.0) | Distribution shape can change, rank ordering locked |
| Sample once, reuse | Perfect (1.0) | Same value across all periods |
