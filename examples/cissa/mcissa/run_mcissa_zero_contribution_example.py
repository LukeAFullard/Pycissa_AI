import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure pycissa is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from pycissa.processing.mcissa.mcissa import MCissa

def main():
    print("Generating M-CiSSA Zero Contribution Independent Signal example...")
    np.random.seed(42)
    T = 200
    t = np.arange(1, T + 1)

    # 1. Base target signals
    trend = 0.05 * t
    periodic = 2 * np.sin(2 * np.pi * t / 12)

    # Mixed signals we want to analyze (e.g. Channel 1 and 2 share components)
    x1 = trend + periodic + np.random.normal(0, 0.1, T)
    x2 = trend + periodic * 1.5 + np.random.normal(0, 0.1, T)

    # 2. Totally independent signal (e.g. Channel 3)
    # Different frequency and no trend
    independent_periodic = 3 * np.sin(2 * np.pi * t / 7)
    x3 = independent_periodic + np.random.normal(0, 0.1, T)

    # 3. Create a multivariate dataset
    X = np.column_stack((x1, x2, x3))

    # 4. Fit MCissa
    L = 24
    print(f"Fitting MCissa with L={L}...")
    mcissa = MCissa(t=t, x=X)
    mcissa.fit(L=L, extension_type='NoExt')
    Z_stacked = mcissa.Z_stacked

    # Group by variance and identify the component most correlated with independent_periodic
    M = 3
    variances = [np.sum([np.var(Z_stacked[:, m, i]) for m in range(M)]) for i in range(Z_stacked.shape[2])]
    sorted_indices = np.argsort(variances)[::-1]

    best_corr = 0
    indep_comp_idx = -1
    for idx in sorted_indices[:5]:
        comp_x3 = Z_stacked[:, 2, idx] # Check correlation with the independent signal channel
        corr = abs(np.corrcoef(comp_x3, independent_periodic)[0, 1])
        if corr > best_corr:
            best_corr = corr
            indep_comp_idx = idx

    # If the frequency is spread out, check combinations. The dominant period often combines frequencies.
    combined_comp = Z_stacked[:, 2, sorted_indices[2]] + Z_stacked[:, 2, sorted_indices[3]] + Z_stacked[:, 2, sorted_indices[4]]
    combined_corr = abs(np.corrcoef(combined_comp, independent_periodic)[0, 1])

    if combined_corr > best_corr:
        best_corr = combined_corr
        print(f"Isolated the independent signal utilizing combined components. Correlation: {best_corr:.4f}")
    else:
        print(f"Isolated the independent signal. Correlation: {best_corr:.4f}")

    # The contribution of this independent component to Channel 1 and 2 should be virtually zero
    indep_comp_in_x1 = Z_stacked[:, 0, indep_comp_idx]
    indep_comp_in_x2 = Z_stacked[:, 1, indep_comp_idx]

    max_contrib_x1 = np.max(np.abs(indep_comp_in_x1))
    max_contrib_x2 = np.max(np.abs(indep_comp_in_x2))

    print(f"Maximum cross-bleed amplitude into Target Channel 1: {max_contrib_x1:.4f}")
    print(f"Maximum cross-bleed amplitude into Target Channel 2: {max_contrib_x2:.4f}")

    # 5. Plot the results
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(x1, label="Channel 1 (Mixed Target)", color="black", alpha=0.5)
    axes[0].plot(x2, label="Channel 2 (Mixed Target)", color="gray", alpha=0.5)
    axes[0].set_title("Target Channels (Variables 0 and 1)")
    axes[0].legend(loc="upper right")

    axes[1].plot(x3, label="Channel 3 (Independent Signal)", color="green", linewidth=2)
    axes[1].plot(Z_stacked[:, 2, indep_comp_idx], label="Extracted Component from Channel 3", color="blue", linestyle="--", linewidth=2)
    axes[1].set_title("Independent Channel and Extracted Signal")
    axes[1].legend(loc="upper right")

    axes[2].plot(indep_comp_in_x1, label=f"Leakage into Channel 1 (Max: {max_contrib_x1:.3f})", color="red")
    axes[2].plot(indep_comp_in_x2, label=f"Leakage into Channel 2 (Max: {max_contrib_x2:.3f})", color="orange")
    axes[2].set_title("Cross-Channel Bleed (Zero Contribution)")
    axes[2].legend(loc="upper right")
    axes[2].set_ylim(-3, 3)

    plt.tight_layout()
    plt.savefig("mcissa_zero_contribution.png")
    print("Saved 'mcissa_zero_contribution.png'")

    print("Zero Contribution Example generation complete.")

if __name__ == "__main__":
    main()
