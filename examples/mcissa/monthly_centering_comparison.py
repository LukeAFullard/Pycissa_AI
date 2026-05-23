import numpy as np
import matplotlib.pyplot as plt
from pycissa.processing.mcissa.mcissa import MCissa
from pycissa.preprocessing.gap_fill.uneven_gap_filling import fill_uneven_timeseries
import os
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Simulate 3 years of "monthly" data.
# The true grid is mid-month: e.g., t=15.2, 45.6, 76.0 etc.
true_t = np.arange(15.2, 365.25 * 3, 365.25 / 12)

# The observed times are randomly jittered around the true times
# Randomly sampled somewhere between day 1 and 28 of the month.
jitter = np.random.uniform(-14, 14, size=len(true_t))
observed_t = true_t + jitter

# Underlying dynamics (Two channels)
# Channel 0: Trend + Annual + Noise
# Channel 1: Related Trend + Annual + Semi-annual + Noise
def generate_signals(t):
    trend1 = 0.01 * t
    trend2 = 0.012 * t
    annual1 = 5 * np.sin(2 * np.pi * t / 365.25)
    annual2 = 4 * np.sin(2 * np.pi * t / 365.25 + np.pi/4) # phase shifted
    semi1 = 0
    semi2 = 2 * np.sin(2 * np.pi * t / (365.25/2))

    noise1 = np.random.normal(0, 0.5, len(t))
    noise2 = np.random.normal(0, 0.5, len(t))

    return trend1 + annual1 + semi1 + noise1, trend2 + annual2 + semi2 + noise2

true_x1, true_x2 = generate_signals(true_t)
obs_x1, obs_x2 = generate_signals(observed_t)

true_x = np.column_stack((true_x1, true_x2))
obs_x = np.column_stack((obs_x1, obs_x2))

# Introduce gaps (missed a couple of months entirely)
gap_indices = [5, 6, 18, 19, 20]
obs_x[gap_indices, :] = np.nan

print("--- Monthly Centering Comparison ---")
print(f"Number of observations: {len(observed_t)}")
print(f"Number of missing data points (gaps): {len(gap_indices)}")

# True target grid will be roughly standard 30.4 days apart starting near 15
dt = 365.25 / 12

print("\n--- Running CiSSA (Independent Univariate) Centering ---")
mc_cissa = MCissa(observed_t, obs_x)
mc_cissa.pre_fill_uneven_timeseries(dt=dt, L_values=[12], gap_threshold=20, test_number=0, multivariate=False, plot=False)
t_cissa = mc_cissa.t
x_cissa = mc_cissa.x

print("\n--- Running M-CiSSA (Joint Multivariate) Centering ---")
mc_mcissa = MCissa(observed_t, obs_x)
mc_mcissa.pre_fill_uneven_timeseries(dt=dt, L_values=[12], gap_threshold=20, test_number=0, multivariate=True, plot=False)
t_mcissa = mc_mcissa.t
x_mcissa = mc_mcissa.x

# Function to compute metrics where we actually have a target
def compute_metrics(true, pred):
    valid = ~np.isnan(true) & ~np.isnan(pred)
    mse = np.mean((true[valid] - pred[valid])**2)
    rmse = np.sqrt(mse)
    corr = np.corrcoef(true[valid], pred[valid])[0, 1]
    return rmse, corr

# Evaluate on the true non-noisy target grid
# We need the pure signal on the generated t_mcissa grid.
# For true comparison, let's regenerate the clean true signal on the t_mcissa grid
clean_true_x1 = 0.01 * t_mcissa + 5 * np.sin(2 * np.pi * t_mcissa / 365.25)
clean_true_x2 = 0.012 * t_mcissa + 4 * np.sin(2 * np.pi * t_mcissa / 365.25 + np.pi/4) + 2 * np.sin(2 * np.pi * t_mcissa / (365.25/2))
clean_true_x = np.column_stack((clean_true_x1, clean_true_x2))

rmse_c0, corr_c0 = compute_metrics(clean_true_x[:,0], x_cissa[:,0])
rmse_c1, corr_c1 = compute_metrics(clean_true_x[:,1], x_cissa[:,1])

rmse_m0, corr_m0 = compute_metrics(clean_true_x[:,0], x_mcissa[:,0])
rmse_m1, corr_m1 = compute_metrics(clean_true_x[:,1], x_mcissa[:,1])

print("\n--- Results against Underlying True Dynamics ---")
print("Channel 0 (Trend + Annual):")
print(f"  CiSSA   - RMSE: {rmse_c0:.4f}, Correlation: {corr_c0:.4f}")
print(f"  M-CiSSA - RMSE: {rmse_m0:.4f}, Correlation: {corr_m0:.4f}")

print("\nChannel 1 (Trend + Annual + Semi-Annual):")
print(f"  CiSSA   - RMSE: {rmse_c1:.4f}, Correlation: {corr_c1:.4f}")
print(f"  M-CiSSA - RMSE: {rmse_m1:.4f}, Correlation: {corr_m1:.4f}")

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

for m in range(2):
    ax = axes[m]

    # Target pure truth
    ax.plot(t_mcissa, clean_true_x[:,m], 'k-', label='Underlying Truth', alpha=0.5, linewidth=3)

    # Observed jittery points
    ax.scatter(observed_t, obs_x[:,m], color='red', marker='x', label='Observed (Jittery & Gappy)', s=50)

    # Predictions
    ax.plot(t_cissa, x_cissa[:,m], 'b--', label='CiSSA (Univariate) Centered')
    ax.plot(t_mcissa, x_mcissa[:,m], 'g.-', label='M-CiSSA (Multivariate) Centered', markersize=8)

    ax.set_title(f'Channel {m} Monthly Centering')
    ax.set_xlabel('Time (Days)')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
os.makedirs('examples/mcissa', exist_ok=True)
plt.savefig('examples/mcissa/monthly_centering_comparison.png')
print("\nPlot saved to examples/mcissa/monthly_centering_comparison.png")
