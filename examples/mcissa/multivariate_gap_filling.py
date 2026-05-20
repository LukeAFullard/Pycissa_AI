import numpy as np
import matplotlib.pyplot as plt
from pycissa.processing.mcissa.mcissa import MCissa
from pycissa.processing.cissa.cissa import Cissa

np.random.seed(42)

# Create a highly correlated multivariate dataset
t = np.arange(0, 100, 1)

# Base signal
base_signal = np.sin(2 * np.pi * t / 20)

# Channel 1: Base signal + noise
c1 = base_signal + 0.1 * np.random.randn(len(t))

# Channel 2: Base signal shifted + trend + noise
c2 = np.cos(2 * np.pi * t / 20) + 0.05 * t + 0.1 * np.random.randn(len(t))

# Channel 3: Base signal highly correlated to c1 but slightly larger amplitude
c3 = 1.5 * base_signal + 0.1 * np.random.randn(len(t))

x_full = np.column_stack([c1, c2, c3])

# Introduce a large gap in Channel 1 (where Univariate fails but Multi uses C2/C3 to reconstruct)
x_gaps = x_full.copy()
x_gaps[40:70, 0] = np.nan

# 1. Fill using Univariate CiSSA independently on each channel
x_uni_filled = np.zeros_like(x_full)
print("Running Univariate Gap Filling...")
for i in range(x_gaps.shape[1]):
    if np.any(np.isnan(x_gaps[:, i])):
        model_uni = Cissa(t, x_gaps[:, i].copy())
        # Use component selection to make sure it converges
        model_uni.pre_fill_gaps(L=40, estimate_error=False, verbose=False, component_selection_method='drop_smallest_proportion', eigenvalue_proportion=0.9)
        x_uni_filled[:, i] = model_uni.x
    else:
        x_uni_filled[:, i] = x_gaps[:, i]

# 2. Fill using Multivariate CiSSA jointly
print("Running Multivariate Gap Filling...")
model_multi = MCissa(t, x_gaps.copy())
model_multi.pre_fill_gaps(L=40, estimate_error=False, verbose=False, component_selection_method='drop_smallest_proportion', eigenvalue_proportion=0.9)
x_multi_filled = model_multi.x

# Calculate RMSE across the gap region
gap_mask = np.isnan(x_gaps)
rmse_uni = np.sqrt(np.mean((x_full[gap_mask] - x_uni_filled[gap_mask])**2))
rmse_multi = np.sqrt(np.mean((x_full[gap_mask] - x_multi_filled[gap_mask])**2))

print(f"\n--- Gap Imputation RMSE ---")
print(f"Univariate independent fill: {rmse_uni:.4f}")
print(f"Multivariate joint fill:     {rmse_multi:.4f}")

# Plotting the comparison
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

for i in range(3):
    axes[i].plot(t, x_full[:, i], 'k-', alpha=0.3, label='Original Complete')
    axes[i].plot(t, x_gaps[:, i], 'ko', markersize=3, label='Data with Gaps')

    if i == 0:
        gap_idx = slice(40, 70)
        axes[i].plot(t[gap_idx], x_uni_filled[gap_idx, i], 'r--', label=f'Univariate Fill')
        axes[i].plot(t[gap_idx], x_multi_filled[gap_idx, i], 'g-', label=f'Multivariate Fill')

    axes[i].set_ylabel(f'Channel {i+1}')
    if i == 0:
        axes[i].legend(loc='upper right', fontsize=8)

axes[2].set_xlabel('Time')
plt.suptitle('Comparison of Univariate vs Multivariate Gap Filling\n(Multivariate leverages complete cross-channels)', y=0.98)
plt.tight_layout()
plt.savefig('examples/mcissa/gap_fill_comparison.png')
print("Saved comparison plot to examples/mcissa/gap_fill_comparison.png")
