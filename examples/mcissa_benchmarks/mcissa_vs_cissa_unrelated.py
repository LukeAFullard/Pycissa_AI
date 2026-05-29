import numpy as np
import matplotlib.pyplot as plt
from pycissa.processing.cissa.cissa import Cissa
from pycissa.processing.mcissa.mcissa import MCissa

# Generate a time series with trend, periodic, and noise
np.random.seed(42)
t = np.arange(0, 1000)
trend = 0.005 * t
periodic = 5 * np.sin(2 * np.pi * t / 100) + 2 * np.cos(2 * np.pi * t / 50)
noise = np.random.normal(0, 1, len(t))
x = trend + periodic + noise

# Add an unrelated time series
unrelated_periodic = 3 * np.sin(2 * np.pi * t / 73)
unrelated_noise = np.random.normal(0, 1.5, len(t))
x2 = unrelated_periodic + unrelated_noise

# Combine into a multivariate time series
X = np.column_stack((x, x2))

# 1. Apply Cissa to the first time series
cissa = Cissa(t, x)
L = 200
cissa.fit(L)
cissa.post_run_monte_carlo_analysis(alpha=0.01)
cissa.post_group_components()
cissa_trend = cissa.x_trend
cissa_periodic = cissa.x_periodic
cissa_noise = cissa.x_noise

cissa_periodic_freqs = cissa.results['cissa']['noise component tests']['periodic_index']

# 2. Apply MCissa to the multivariate time series
mcissa = MCissa(t, X)
mcissa.fit(L)
mcissa.post_run_monte_carlo_analysis(alpha=0.01)
mcissa.post_group_components()

# MCissa grouping results are structured like the original time series: (T, M)
mcissa_trend = mcissa.x_trend[:, 0]
mcissa_periodic = mcissa.x_periodic[:, 0]
mcissa_noise = mcissa.x_noise[:, 0]

mcissa_periodic_freqs = mcissa.results['mcissa']['noise component tests']['periodic_index']

# 3. Quantify the differences
diff_trend = np.mean(np.abs(cissa_trend - mcissa_trend))
diff_periodic = np.mean(np.abs(cissa_periodic - mcissa_periodic))
diff_noise = np.mean(np.abs(cissa_noise - mcissa_noise))

print("=== Comparison between Cissa and MCissa ===")
print("When applying MCissa to a multivariate time series consisting of a target signal and an unrelated signal,")
print("the extraction of the target signal components differs slightly from univariate Cissa because MCissa considers joint spatial significance.")
print("")
print(f"Mean Absolute Difference in Trend:    {diff_trend:.4e}")
print(f"Mean Absolute Difference in Periodic: {diff_periodic:.4f}")
print(f"Mean Absolute Difference in Noise:    {diff_noise:.4f}")

print(f"\nCissa periodic frequencies:  {cissa_periodic_freqs}")
print(f"MCissa periodic frequencies: {mcissa_periodic_freqs}")

# Plotting the results
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

axs[0].plot(t, x, label='Original Target Series', color='k', alpha=0.5)
axs[0].plot(t, x2, label='Unrelated Added Series', color='gray', alpha=0.3)
axs[0].set_title('Original Time Series')
axs[0].legend()

axs[1].plot(t, cissa_trend, label='Cissa Trend', linestyle='-')
axs[1].plot(t, mcissa_trend, label='MCissa Trend (Channel 0)', linestyle='--')
axs[1].set_title(f'Trend (MAE: {diff_trend:.4e})')
axs[1].legend()

axs[2].plot(t, cissa_periodic, label='Cissa Periodic', linestyle='-')
axs[2].plot(t, mcissa_periodic, label='MCissa Periodic (Channel 0)', linestyle='--')
axs[2].set_title(f'Periodic (MAE: {diff_periodic:.4f})')
axs[2].legend()

axs[3].plot(t, cissa_noise, label='Cissa Noise', linestyle='-')
axs[3].plot(t, mcissa_noise, label='MCissa Noise (Channel 0)', linestyle='--')
axs[3].set_title(f'Noise (MAE: {diff_noise:.4f})')
axs[3].legend()

plt.tight_layout()
plt.savefig('examples/mcissa_benchmarks/mcissa_vs_cissa_unrelated.png')
print("Saved plot to examples/mcissa_benchmarks/mcissa_vs_cissa_unrelated.png")
