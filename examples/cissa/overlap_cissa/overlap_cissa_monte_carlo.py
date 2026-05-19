import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add root directory to python path if not installed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from pycissa.processing.cissa.overlap_cissa import OverlapCissa

def run_monte_carlo_example():
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

    # Define parameters for OverlapCissa
    L = 100
    q = 200
    L_bar = 25
    Z_len = q + 2 * L_bar # 250

    # 2. Process with OverlapCissa
    print("Running OverlapCissa...")
    ocissa = OverlapCissa(t, x, Z=Z_len, q=q, L=L)
    ocissa.fit()

    # 3. Apply Monte Carlo surrogates to find significant components
    print("Running Monte Carlo significance testing...")
    # alpha=0.05 gives 95% confidence interval
    ocissa.post_run_monte_carlo_analysis(alpha=0.05, K_surrogates=5, surrogates='random_permutation')

    # Group components using the results of the monte carlo analysis
    # This automatically sums the significant periodic components and noise components.
    print("Grouping components based on Monte Carlo results...")
    ocissa.post_group_components(grouping_type='monte_carlo', plot_result=False)

    # 4. Plot results
    print("Plotting results...")
    plt.figure(figsize=(14, 7))
    plt.plot(t, x, 'k.', alpha=0.3, label='Original Data (with noise)')

    # Plot the reconstructed main signal (trend + periodic) based on Monte Carlo
    x_mc_recon = ocissa.x_trend + ocissa.x_periodic
    plt.plot(t, x_mc_recon, 'r-', label='OverlapCissa Monte Carlo Reconstruction', linewidth=2)

    # Mark structural break
    plt.axvline(750, color='g', linestyle=':', linewidth=2, label='Structural Break')

    plt.title("OverlapCissa Reconstruction using Monte Carlo Significant Components", fontsize=14)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    output_path = os.path.join(os.path.dirname(__file__), 'overlap_cissa_monte_carlo.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Monte Carlo plot saved to: {output_path}")

if __name__ == "__main__":
    run_monte_carlo_example()
