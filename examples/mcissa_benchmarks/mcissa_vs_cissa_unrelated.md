# M-CiSSA vs CiSSA with Unrelated Signals

This example demonstrates an experiment where we apply univariate CiSSA to a target time series to extract its trend, periodic components, and noise. We then construct a multivariate time series by pairing the target time series with a completely unrelated time series and apply M-CiSSA to this combination.

## Motivation

The goal is to see if M-CiSSA extracts the same components for the target time series when it is processed alongside an unrelated signal, and to quantify any differences.

## Findings

Because M-CiSSA decomposes the signal using the joint cross-spectral density matrix, the significance of frequency components in the Monte Carlo test is evaluated across all channels simultaneously.

If an unrelated signal has a strong periodic component at a specific frequency, M-CiSSA may flag that frequency as globally significant. Consequently, that frequency component will be extracted from *all* channels and included in their `x_periodic` reconstructions, even if its amplitude in the target channel is extremely small (i.e., just noise that happened to align with that frequency).

In our experiment:
- The target time series has periodicities at $T=100$ (index 2) and $T=50$ (index 4).
- The unrelated time series has a periodicity at $T \approx 73$ (index 3).
- Univariate CiSSA correctly identifies indices `[2, 4]` as significant for the target series.
- M-CiSSA identifies indices `[2, 3, 4]` as globally significant because of the strong presence of index 3 in the unrelated series.

### Quantifying the Difference

- **Trend MAE**: ~0.00 (Trend extraction is robust and identical)
- **Periodic MAE**: ~0.0789 (The target's periodic reconstruction now includes a small amount of noise at frequency index 3)
- **Noise MAE**: ~0.0789 (That same small amount of noise was removed from the noise component)

The difference in the reconstructed periodic and noise components is minimal because M-CiSSA scales the spatial eigenvector weights according to the covariance. Thus, the amplitude of the incorrectly grouped "unrelated" periodicity in the target channel remains very small.

## How to run

```bash
poetry run python examples/mcissa_benchmarks/mcissa_vs_cissa_unrelated.py
```
