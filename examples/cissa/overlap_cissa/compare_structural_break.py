import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add root directory to python path if not installed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from pycissa.processing.cissa.cissa import Cissa
from pycissa.processing.cissa.overlap_cissa import OverlapCissa

def run_comparison():
    np.random.seed(42)

    # 1. Generate time series with a structural break
    N = 1000
    t = np.arange(N)

    # First 3/4: freq = 50, amplitude = 1
    x1 = np.sin(2 * np.pi * t[:750] / 50)
    # Last 1/4: freq = 20, amplitude = 2.5
    x2 = 2.5 * np.sin(2 * np.pi * t[750:] / 20)

    # Add noise
    x = np.concatenate([x1, x2]) + np.random.normal(0, 0.5, N)

    # Define parameters
    L = 100

    # 2. Process with Standard Cissa
    print("Running standard Cissa...")
    cissa = Cissa(t, x)
    cissa.fit(L=L)

    # Group components (drop smallest N to separate signal from noise)
    # The signal is very distinct, so dropping the tail end is sufficient
    cissa.post_group_components(grouping_type='smallest_n', number_of_groups_to_drop=L//2 - 5, include_trend=True, plot_result=False)
    x_cissa_recon = cissa.x_trend + cissa.x_periodic

    # 3. Process with OverlapCissa
    print("Running OverlapCissa...")
    # Z = q + 2 * L_bar. Z must be > 2*L for cissa fit to work.
    # So q + 2*L_bar > 200.
    q = 200
    L_bar = 25
    Z_len = q + 2 * L_bar # 250

    ocissa = OverlapCissa(t, x, Z=Z_len, q=q, L=L)
    ocissa.fit()

    # Reconstruct OverlapCissa
    # Since we don't have the full post_grouping output formatted for the global Z,
    # we can do a simple top-component sum for reconstruction of the main signal.
    # The main signal + trend will be in the top few components.
    x_ocissa_recon = np.sum(ocissa.Z[:, :10], axis=1)

    # 4. Plot results
    print("Plotting results...")
    plt.figure(figsize=(14, 7))
    plt.plot(t, x, 'k.', alpha=0.3, label='Original Data (with noise)')
    plt.plot(t, x_cissa_recon, 'b-', label='Standard Cissa Reconstruction', linewidth=2)
    plt.plot(t, x_ocissa_recon, 'r--', label='OverlapCissa Reconstruction', linewidth=2)

    # Mark structural break
    plt.axvline(750, color='g', linestyle=':', linewidth=2, label='Structural Break')

    plt.title("Cissa vs OverlapCissa on Series with Structural Break", fontsize=14)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    output_path = os.path.join(os.path.dirname(__file__), 'structural_break_comparison.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Comparison plot saved to: {output_path}")

if __name__ == "__main__":
    run_comparison()
