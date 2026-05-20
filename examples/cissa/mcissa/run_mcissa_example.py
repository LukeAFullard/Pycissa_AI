import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure pycissa is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from pycissa.processing.mcissa.mcissa import MCissa

def main():
    print("Generating M-CiSSA verification example...")
    np.random.seed(42)
    N = 300
    M = 3 # 3 variables

    # Generate synthetic subsignals for variable 0 (Trend, Low Freq, High Freq, Noise)
    # The M-CiSSA test constructs signals such that they are distinct in frequency

    # 1. Trend
    trend_0 = np.linspace(0, 5, N)
    trend_1 = np.linspace(0, 15, N)
    trend_2 = np.linspace(0, -5, N)
    sub_trend = np.column_stack((trend_0, trend_1, trend_2))

    # 2. Low Frequency Oscillation (Shared frequency, different amplitudes and phases)
    t = np.arange(N)
    low_freq_0 = 2.0 * np.sin(2 * np.pi * t / 50 + 0.0)
    low_freq_1 = 3.0 * np.sin(2 * np.pi * t / 50 + np.pi / 4)
    low_freq_2 = 1.0 * np.sin(2 * np.pi * t / 50 + np.pi / 2)
    sub_low = np.column_stack((low_freq_0, low_freq_1, low_freq_2))

    # 3. High Frequency Oscillation (Shared frequency, different amplitudes and phases)
    high_freq_0 = 0.5 * np.sin(2 * np.pi * t / 5 + 0.0)
    high_freq_1 = 1.3 * np.sin(2 * np.pi * t / 5 + np.pi / 3)
    high_freq_2 = 0.8 * np.sin(2 * np.pi * t / 5 + np.pi / 1.5)
    sub_high = np.column_stack((high_freq_0, high_freq_1, high_freq_2))

    # Combine signals (we omit noise here to cleanly show reconstruction, or we can include a small amount)
    X = sub_trend + sub_low + sub_high

    # Define parameters
    L = 100

    # Initialize and fit MCissa
    print(f"Fitting MCissa with L={L}...")
    t = np.arange(N)
    mcissa = MCissa(t=t, x=X)
    mcissa.fit(L=L)

    # We want to reconstruct the components. Since this is an example, let's group by eigenvalues.
    # Groupings would normally be done dynamically or visually. Here we know the structures.
    # We'll just plot the first 6 components using the built in plotting method.

    print("Saving standard components plot...")
    mcissa.plot_components(num_components=6)
    plt.savefig("mcissa_components.png")
    plt.close()

    # Now let's explicitly reconstruct and verify the error against the original data
    print("Verifying perfect reconstruction...")
    # Reconstruct the entire signal using all L*M components
    # the mcissa fit produces `.Z_stacked` which contains all the components.
    # shape is (T, M, L*M)
    X_recon = np.sum(mcissa.Z_stacked, axis=2)

    # We will compute the error up to original signal length N
    error = X - X_recon[:N]
    max_error = np.max(np.abs(error))
    print(f"Max reconstruction error: {max_error}")

    # Plot Original vs Reconstruction vs Error for all variables
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))

    for m in range(M):
        # Original
        axes[0, m].plot(X[:, m], label=f"Original (Var {m})", color='black')
        axes[0, m].set_title(f"Original Mixed Signal (Variable {m})")
        axes[0, m].legend()

        # Reconstructed
        axes[1, m].plot(X_recon[:, m], label=f"Reconstructed (Var {m})", color='blue', linestyle='--')
        axes[1, m].set_title(f"Full Reconstruction (Variable {m})")
        axes[1, m].legend()

        # Error
        axes[2, m].plot(error[:, m], label=f"Error (Var {m})", color='red')
        axes[2, m].set_title(f"Reconstruction Error (Variable {m})")
        axes[2, m].legend()

    plt.tight_layout()
    plt.savefig("mcissa_reconstruction_error.png")
    print("Saved 'mcissa_reconstruction_error.png'")

    print("Example generation complete.")

if __name__ == "__main__":
    main()
