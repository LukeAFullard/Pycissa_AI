import numpy as np
import matplotlib.pyplot as plt
import warnings
from pycissa.preprocessing.gap_fill.uneven_gap_filling import m_fill_uneven_timeseries

def main():
    # 1. Generate Synthetic Data
    np.random.seed(42)
    t = np.sort(np.random.uniform(0, 100, 200))
    x1 = np.sin(2 * np.pi * t / 20) + np.random.normal(0, 0.5, 200)
    x2 = np.cos(2 * np.pi * t / 20) + np.random.normal(0, 0.5, 200)
    x = np.column_stack([x1, x2])

    # 2. Punch massive structural gaps to simulate extreme sparsity / sensor failure
    gap_mask = ((t > 15) & (t < 45)) | ((t > 55) & (t < 85))
    t_sparse = t[~gap_mask]
    x_sparse = x[~gap_mask]

    print(f"Original points: {len(t)}, Points after extreme censorship: {len(t_sparse)}")

    # 3. Apply the Multivariate Gap Filler (will trigger fallbacks due to sparsity)
    # We suppress the expected UserWarnings for a clean console output in this example.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = m_fill_uneven_timeseries(
            t=t_sparse,
            x=x_sparse,
            L_values=[5, 10, 15],
            dt=1.0,
            gap_threshold=5.0,
            interp_method='pchip',
            r2_warning_threshold=0.9, # Artificially high to force fallbacks
            plot=False # We handle plotting manually below
        )

        fallback_warnings = [warn.message for warn in w if "Poor fit detected" in str(warn.message)]
        if fallback_warnings:
            print(f"Fallback mechanism issued warning: {fallback_warnings[0]}")

    print(f"Algorithm finalized on L={res['best_L']} with R2={res['r2']:.4f}")

    # 4. Visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    t_even = res['t_even']
    x_filled = res['x_even_filled']

    for i in range(2):
        # Ground Truth
        axes[i].plot(t, x[:, i], 'k-', alpha=0.2, label='True Hidden Dynamics')
        # What the sensor actually captured
        axes[i].plot(t_sparse, x_sparse[:, i], 'ro', markersize=6, label='Sparse Sparse Data')
        # The algorithmic reconstruction
        axes[i].plot(t_even, x_filled[:, i], 'b--', linewidth=2, label='M-CiSSA Reconstruction (Post-Fallback)')

        axes[i].set_title(f'Channel {i+1}')
        axes[i].legend()
        axes[i].grid(True)

    plt.suptitle("M-CiSSA Gap Filling on Extremely Sparse Data (Fallback Mechanism)")
    plt.tight_layout()
    plt.savefig('extreme_sparsity_fallback.png', dpi=300)
    print("Plot saved to extreme_sparsity_fallback.png")

if __name__ == "__main__":
    main()
