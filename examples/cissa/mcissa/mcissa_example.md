# M-CiSSA (Multivariate Circulant Singular Spectrum Analysis) Example

This example demonstrates how to use the `MCissa` class to analyze and decompose multivariate time series data. We generate a synthetic dataset with three variables, each composed of a trend, a low-frequency oscillation, and a high-frequency oscillation.

## Running the Example

To run the verification script, execute the following from this directory:

```bash
python run_mcissa_example.py
```

## Explanation of the Code

First, we generate the synthetic data:

```python
import numpy as np
import matplotlib.pyplot as plt
from pycissa.processing.mcissa.mcissa import MCissa

np.random.seed(42)
N = 300
M = 3 # 3 variables

# 1. Trend
trend_0 = np.linspace(0, 5, N)
trend_1 = np.linspace(0, 15, N)
trend_2 = np.linspace(0, -5, N)
sub_trend = np.column_stack((trend_0, trend_1, trend_2))

# 2. Low Frequency Oscillation
t = np.arange(N)
low_freq_0 = 2.0 * np.sin(2 * np.pi * t / 50)
low_freq_1 = 3.0 * np.sin(2 * np.pi * t / 20)
low_freq_2 = 1.0 * np.sin(2 * np.pi * t / 80)
sub_low = np.column_stack((low_freq_0, low_freq_1, low_freq_2))

# 3. High Frequency Oscillation
high_freq_0 = 0.5 * np.sin(2 * np.pi * t / 5)
high_freq_1 = 1.3 * np.sin(2 * np.pi * t / 7)
high_freq_2 = 0.8 * np.sin(2 * np.pi * t / 3)
sub_high = np.column_stack((high_freq_0, high_freq_1, high_freq_2))

# Combine signals
X = sub_trend + sub_low + sub_high
```

We then initialize and fit the `MCissa` class with window length `L=100`:

```python
L = 100
mcissa = MCissa(t=t, x=X)
mcissa.fit(L=L)
```

We can plot the top components (grouped by variance across all variables) directly using the built-in `plot_components` method:

```python
mcissa.plot_components(num_components=6)
plt.savefig("mcissa_components.png")
```

The resulting `mcissa_components.png` file will show a grid of the original data and the top components isolated.

Finally, we can verify that summing all the components perfectly reconstructs the original signal (a key property of SSA techniques):

```python
# Reconstruct the entire signal using all L*M components
# the mcissa fit produces `.Z_stacked` which contains all the components.
# shape is (T, M, L*M)
X_recon = np.sum(mcissa.Z_stacked, axis=2)

error = X - X_recon[:N]
max_error = np.max(np.abs(error))
print(f"Max reconstruction error: {max_error}")

# Plot Original vs Reconstruction vs Error for all variables
fig, axes = plt.subplots(3, 3, figsize=(18, 12))

for m in range(M):
    # Original
    axes[0, m].plot(X[:, m], label=f"Original (Var {m})", color='black')
    axes[0, m].set_title(f"Original Mixed Signal (Variable {m})")
    axes[0, m].legend()

    # Reconstructed
    axes[1, m].plot(X_recon[:, m], label=f"Reconstructed (Var {m})", color='blue', linestyle='--')
    axes[1, m].set_title(f"Full Reconstruction (Variable {m})")
    axes[1, m].legend()

    # Error
    axes[2, m].plot(error[:, m], label=f"Error (Var {m})", color='red')
    axes[2, m].set_title(f"Reconstruction Error (Variable {m})")
    axes[2, m].legend()

plt.tight_layout()
plt.savefig("mcissa_reconstruction_error.png")
```

The output confirms the reconstruction is perfect (within floating point precision limits):

```
Max reconstruction error: 5.684341886080802e-14
```

## Generated Plots

When you run the code, it will produce the standard component plots, isolating the trends, and various frequencies from each other, which you can see in `mcissa_components.png`.

It will also produce `mcissa_reconstruction_error.png` which shows a 3x3 grid comparing the original signal, the reconstructed signal, and the error for all three variables, verifying that when we sum all components, the original series is recovered perfectly.