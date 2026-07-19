import numpy as np
import matplotlib.pyplot as plt
from pycissa import Cissa
from pycissa.processing.mcissa.mcissa import MCissa

# --- 1. Generate Synthetic Data ---
# We simulate a 100-day time series with an underlying periodic signal + trend
np.random.seed(42)
t_true = np.arange(0, 100, 1.0)
true_trend = 0.5 * t_true
true_periodic = 10 * np.sin(2 * np.pi * t_true / 20)
y_true = true_trend + true_periodic

# Add some noise
y_noisy = y_true + np.random.normal(0, 2, len(t_true))

# Simulate uneven sampling (randomly drop 30% of points)
drop_indices = np.random.choice(len(t_true), size=int(0.3 * len(t_true)), replace=False)
mask = np.ones(len(t_true), dtype=bool)
mask[drop_indices] = False

t_uneven = t_true[mask]
y_uneven = y_noisy[mask]

# Simulate a large gap between day 40 and 60
gap_mask = (t_uneven < 40) | (t_uneven > 60)
t_uneven = t_uneven[gap_mask]
y_uneven = y_uneven[gap_mask]

# --- 2. Univariate Gap Filling ---
cissa_model = Cissa(t_uneven, y_uneven)
print("Running Univariate CiSSA Gap Filling...")
cissa_model.pre_fill_uneven_timeseries(
    L_values=[20],
    dt=1.0,
    gap_threshold=2.5, # Any gap > 2.5 days is considered missing data and masked
    update_state=True,
    plot=False, # We will plot our own comparison
    outliers=['nan_only', None]
)

# Extract reconstructed grid
t_even_cissa = cissa_model.t
y_filled_cissa = cissa_model.x

# Calculate metrics strictly for the large gap region (40-60)
gap_indices_cissa = (t_even_cissa >= 40) & (t_even_cissa <= 60)
gap_true_cissa = true_trend[40:61] + true_periodic[40:61] # Corresponds to t=40..60
gap_pred_cissa = y_filled_cissa[gap_indices_cissa]

rmse_cissa = np.sqrt(np.mean((gap_true_cissa - gap_pred_cissa)**2))
ss_res_cissa = np.sum((gap_true_cissa - gap_pred_cissa)**2)
ss_tot_cissa = np.sum((gap_true_cissa - np.mean(gap_true_cissa))**2)
r2_cissa = 1 - (ss_res_cissa / ss_tot_cissa)

print(f"Univariate - Gap RMSE: {rmse_cissa:.4f}, Gap R2: {r2_cissa:.4f}")

# Plot Univariate
plt.figure(figsize=(12, 6))
plt.plot(t_true, y_true, 'k-', alpha=0.3, label="True Underlying Signal", linewidth=2)
plt.plot(t_uneven, y_uneven, 'bo', label="Measured Uneven Data", markersize=4)
plt.plot(t_even_cissa, y_filled_cissa, 'r--', label=f"CiSSA Reconstruction (Gap RMSE={rmse_cissa:.2f})")
plt.axvspan(40, 60, color='gray', alpha=0.2, label="Large Gap Region")
plt.title("Univariate CiSSA: Uneven Data Gap Filling Accuracy")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('examples/uneven_gap_filling_accuracy/univariate_accuracy.png')
plt.close()

# --- 3. Multivariate Gap Filling ---
# Add a second channel that shares the periodic signal but has a different trend and noise
true_trend_2 = -0.2 * t_true + 20
y_true_2 = true_trend_2 + 0.8 * true_periodic
y_noisy_2 = y_true_2 + np.random.normal(0, 3, len(t_true))

# Use exact same uneven sampling and gap for channel 2
y_uneven_2 = y_noisy_2[mask][gap_mask]

# Stack into multivariate shape (T, 2)
Y_multivariate = np.column_stack((y_uneven, y_uneven_2))

mcissa_model = MCissa(t_uneven, Y_multivariate)
print("Running Multivariate M-CiSSA Gap Filling...")
mcissa_model.pre_fill_uneven_timeseries(
    L_values=[20],
    dt=1.0,
    gap_threshold=2.5,
    update_state=True,
    plot=False,
    multivariate=True,
    outliers=['nan_only', None]
)

# Extract reconstructed grid
t_even_mcissa = mcissa_model.t
Y_filled_mcissa = mcissa_model.x

# Calculate metrics strictly for the large gap region (40-60) for Channel 1
gap_pred_mcissa_ch1 = Y_filled_mcissa[gap_indices_cissa, 0]

rmse_mcissa = np.sqrt(np.mean((gap_true_cissa - gap_pred_mcissa_ch1)**2))
ss_res_mcissa = np.sum((gap_true_cissa - gap_pred_mcissa_ch1)**2)
ss_tot_mcissa = np.sum((gap_true_cissa - np.mean(gap_true_cissa))**2)
r2_mcissa = 1 - (ss_res_mcissa / ss_tot_cissa)

print(f"Multivariate - Gap RMSE: {rmse_mcissa:.4f}, Gap R2: {r2_mcissa:.4f}")

# Plot Multivariate (Channel 1 only for direct comparison)
plt.figure(figsize=(12, 6))
plt.plot(t_true, y_true, 'k-', alpha=0.3, label="True Underlying Signal (Ch 1)", linewidth=2)
plt.plot(t_uneven, Y_multivariate[:, 0], 'bo', label="Measured Uneven Data (Ch 1)", markersize=4)
plt.plot(t_even_mcissa, Y_filled_mcissa[:, 0], 'g--', label=f"M-CiSSA Reconstruction (Gap RMSE={rmse_mcissa:.2f})")
plt.axvspan(40, 60, color='gray', alpha=0.2, label="Large Gap Region")
plt.title("Multivariate M-CiSSA: Joint Channel Gap Filling Accuracy (Channel 1)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('examples/uneven_gap_filling_accuracy/multivariate_accuracy.png')
plt.close()

print("Plots saved.")
