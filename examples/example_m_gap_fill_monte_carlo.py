import numpy as np
import matplotlib.pyplot as plt
from pycissa.processing.mcissa.mcissa import MCissa

def main():
    # 1. Generate Synthetic Multivariate Data
    np.random.seed(42)
    T = 100
    t = np.arange(T)

    # Base signal (sine wave) with different phases/amplitudes for two channels
    s1 = 5.0 * np.sin(2 * np.pi * t / 20)
    s2 = 3.0 * np.sin(2 * np.pi * t / 20 + np.pi/4)

    # Add noise
    n1 = np.random.normal(0, 0.5, T)
    n2 = np.random.normal(0, 0.5, T)

    x1 = s1 + n1
    x2 = s2 + n2

    # Combine into (T, M) array
    x = np.column_stack([x1, x2])

    # Create gaps (mask with NaN)
    x[30:40, 0] = np.nan # 10-point gap in channel 1
    x[60:70, 1] = np.nan # 10-point gap in channel 2

    print("Initializing MCissa with gapped data...")
    model = MCissa(t, x)

    # 2. Perform Gap Filling using true multivariate Monte Carlo
    print("Running multivariate gap filling with Monte Carlo significance testing...")
    # Using small K_surrogates to keep the example fast, but allowing sufficient max_iter to converge
    # We lower test_repeats to 0 to prevent the outer evaluation loop from slowing down the example script too much.
    model.pre_fill_gaps(
        L=20,
        component_selection_method='monte_carlo_significant_components',
        K_surrogates=19, # 19 surrogates for a 0.05 significance level
        alpha=0.05,
        max_iter=50,
        test_repeats=0,
        verbose=True
    )

    x_filled = model.x

    # 3. Plot the results
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for m in range(2):
        ax = axes[m]

        # Original clean signal (for reference)
        orig_s = s1 if m == 0 else s2
        orig_x = x1 if m == 0 else x2

        # Plot full noisy signal as thin gray line
        ax.plot(t, orig_x, color='lightgray', label='Original Noisy Signal (Hidden by gaps)')

        # Plot the data with gaps that was passed to the model
        ax.plot(t, x[:, m], 'o', color='black', markersize=4, label='Input Data (With Gaps)')

        # Plot the gap-filled result
        ax.plot(t, x_filled[:, m], 'r--', linewidth=2, label='Gap-Filled Data')

        ax.set_title(f'Channel {m+1}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.xlabel('Time')
    plt.suptitle('Multivariate Gap Filling using Monte Carlo Significance Testing')
    plt.tight_layout()

    output_file = 'examples/m_gap_fill_monte_carlo_plot.png'
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    main()
