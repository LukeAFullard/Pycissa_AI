import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from pycissa.preprocessing.gap_fill.uneven_gap_filling import fill_uneven_timeseries

def run_jitter_centering_comparison():
    np.random.seed(42)

    # Generate full true signal on an exact monthly grid
    N_months = 400
    t_even = np.arange(15, 15 + N_months * 30, 30.0)

    # Introduce structural break halfway through
    x1 = np.sin(2 * np.pi * t_even[:200] / 365) # Yearly cycle
    x2 = 2 * np.sin(2 * np.pi * t_even[200:] / (365/2)) # 6-month cycle, larger amp
    x_true_even = np.concatenate([x1, x2]) + np.random.normal(0, 0.2, N_months)

    # Simulate "jittery" sampling
    t_uneven = t_even + np.random.uniform(-14, 14, N_months)

    x1_un = np.sin(2 * np.pi * t_uneven[:200] / 365)
    x2_un = 2 * np.sin(2 * np.pi * t_uneven[200:] / (365/2))
    x_uneven = np.concatenate([x1_un, x2_un]) + np.random.normal(0, 0.2, N_months)

    # Randomly drop a few months so there are actually gaps to fill
    keep_prob = 0.85
    mask_keep = np.random.rand(N_months) < keep_prob
    t_uneven = t_uneven[mask_keep]
    x_uneven = x_uneven[mask_keep]

    print("Running standard Cissa to center jittery data onto grid...")
    # gap_threshold < dt prevents valid measurements from being mistakenly labeled as missing.
    gap_threshold = 20.0
    res_standard = fill_uneven_timeseries(
        t=t_uneven,
        x=x_uneven,
        L_values=[12],
        dt=30.0,
        gap_threshold=gap_threshold,
        optimization_metric='rmse',
        plot=False,
        use_cissa_overlap=False
    )

    print("Running OverlapCissa to center jittery data onto grid...")
    res_overlap = fill_uneven_timeseries(
        t=t_uneven,
        x=x_uneven,
        L_values=[12],
        dt=30.0,
        gap_threshold=gap_threshold,
        optimization_metric='rmse',
        plot=False,
        use_cissa_overlap=True,
        q=200,
        L_bar=50
    )

    print(f"Standard Cissa - RMSE vs Back-Interp: {res_standard['rmse']:.4f}, R2: {res_standard['r2']:.4f}")
    print(f"OverlapCissa   - RMSE vs Back-Interp: {res_overlap['rmse']:.4f}, R2: {res_overlap['r2']:.4f}")

    t_eval = res_standard['t_even']
    # Instead of matching index perfectly, we map t_eval strictly back to original t_even by offset logic
    # because t_eval = np.arange(np.nanmin(t_uneven), np.nanmax(t_uneven) + dt, dt)
    # which implies it's shifted based on np.nanmin(t_uneven).

    # We interpolate the results against the pure ground truth t_even targets to find absolute tracking error
    from scipy.interpolate import interp1d
    interp_std = interp1d(res_standard['t_even'], res_standard['x_even_filled'], kind='cubic', fill_value='extrapolate')
    interp_ov = interp1d(res_overlap['t_even'], res_overlap['x_even_filled'], kind='cubic', fill_value='extrapolate')

    gt_eval = x_true_even
    std_eval = interp_std(t_even)
    ov_eval = interp_ov(t_even)

    rmse_std_gt = np.sqrt(np.mean((gt_eval - std_eval)**2))
    rmse_ov_gt = np.sqrt(np.mean((gt_eval - ov_eval)**2))

    print(f"Standard Cissa - True RMSE (vs hidden exact grid): {rmse_std_gt:.4f}")
    print(f"OverlapCissa   - True RMSE (vs hidden exact grid): {rmse_ov_gt:.4f}")

    # Plot
    plt.figure(figsize=(14, 8))

    plt.plot(t_even, x_true_even, 'k-', alpha=0.3, lw=2, label='Hidden Target Grid (True Signal)')
    plt.plot(t_uneven, x_uneven, 'ko', markersize=5, label='Uneven Monthly Samples (Jittered)')

    plt.plot(res_standard['t_even'], res_standard['x_even_filled'], 'bx-', markersize=4, lw=1.5,
             label=f"Standard Cissa Centered Grid")
    plt.plot(res_overlap['t_even'], res_overlap['x_even_filled'], 'r.-', markersize=6, lw=1.5,
             label=f"OverlapCissa Centered Grid")

    plt.axvline(15 + 200*30, color='g', linestyle=':', label='Structural Break')

    plt.title("Centering Jittery Monthly Samples onto an Even Grid: Standard vs OverlapCissa", fontsize=14)
    plt.xlabel("Time (Days)")
    plt.ylabel("Amplitude")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    output_path = os.path.join(os.path.dirname(__file__), 'gap_centering_jitter_comparison.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Comparison plot saved to: {output_path}")

if __name__ == "__main__":
    run_jitter_centering_comparison()
