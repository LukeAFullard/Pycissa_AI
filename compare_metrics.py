import numpy as np
import matplotlib.pyplot as plt
import os
from pycissa.preprocessing.gap_fill.uneven_gap_filling import fill_uneven_timeseries

os.makedirs('examples/cissa/gap_filling', exist_ok=True)

np.random.seed(45)

t_even = np.arange(0, 100, 1.0)
signal_true = 2 * np.sin(2 * np.pi * t_even / 15) + 1.0 * np.cos(2 * np.pi * t_even / 5)
x_true = signal_true + np.random.normal(0, 0.8, len(t_even))

keep_prob = 0.5
mask_random = np.random.rand(len(t_even)) < keep_prob
mask_gap = (t_even > 40) & (t_even < 60)
mask_keep = mask_random & ~mask_gap

t_uneven = t_even[mask_keep]
x_uneven = x_true[mask_keep]

# Use tighter convergence tolerances and increase max iterations
L_values_grid = [10, 15]
eps_values_grid = [0.01, 0.1, 0.5]

res_rmse = fill_uneven_timeseries(
    t=t_uneven,
    x=x_uneven,
    L_values=L_values_grid,
    dt=1.0,
    gap_threshold=2.0,
    eps_values=eps_values_grid,
    interp_method='cubic',
    optimization_metric='rmse',
    estimate_error=False,
    plot=False,
    max_iter=100
)

res_ccc = fill_uneven_timeseries(
    t=t_uneven,
    x=x_uneven,
    L_values=L_values_grid,
    dt=1.0,
    gap_threshold=2.0,
    eps_values=eps_values_grid,
    interp_method='cubic',
    optimization_metric='ccc',
    estimate_error=False,
    plot=False,
    max_iter=100
)

from scipy.interpolate import interp1d
interp_rmse = interp1d(res_rmse['t_even'], res_rmse['x_even_filled'], kind='cubic', fill_value='extrapolate')
x_pred_rmse = interp_rmse(t_even)

interp_ccc = interp1d(res_ccc['t_even'], res_ccc['x_even_filled'], kind='cubic', fill_value='extrapolate')
x_pred_ccc = interp_ccc(t_even)

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(t_even, x_true, 'k-', alpha=0.2, lw=3, label='True Signal + Noise')
ax.plot(t_even, signal_true, 'g-', alpha=0.5, lw=2, label='True Clean Signal')
ax.plot(t_uneven, x_uneven, 'ko', markersize=4, label='Measured Points')

mask_missing = ~mask_keep
ax.plot(t_even[mask_missing], x_pred_rmse[mask_missing], 'b--', lw=2,
        label=f"RMSE Opt (L={res_rmse['best_L']}, eps={res_rmse['best_eps']})")
ax.plot(t_even[mask_missing], x_pred_ccc[mask_missing], 'r:', lw=3,
        label=f"CCC Opt (L={res_ccc['best_L']}, eps={res_ccc['best_eps']})")

ax.set_ylim(np.min(x_true)-1, np.max(x_true)+1)
ax.set_title('Optimization Metric Comparison: RMSE vs CCC (Tighter Convergence)')
ax.set_xlabel('Time')
ax.set_ylabel('Signal Value')
ax.legend()
ax.grid(True)

plt.tight_layout()
fig.savefig('examples/cissa/gap_filling/uneven_gap_filling_ccc_vs_rmse.png')
print("Saved image to examples/cissa/gap_filling/uneven_gap_filling_ccc_vs_rmse.png")

print(f"RMSE Optimization selected L={res_rmse['best_L']}, eps={res_rmse['best_eps']}")
print(f"CCC Optimization selected L={res_ccc['best_L']}, eps={res_ccc['best_eps']}")
