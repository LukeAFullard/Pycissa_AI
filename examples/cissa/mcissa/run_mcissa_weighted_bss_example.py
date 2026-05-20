import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure pycissa is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from pycissa.processing.mcissa.mcissa import MCissa

def main():
    print("Generating M-CiSSA Weighted/Partial BSS example...")
    np.random.seed(42)
    N = 300
    t = np.arange(N)

    # 1. Generate independent pure signals
    s1 = np.linspace(0, 10, N)  # Trend
    s2 = 3.0 * np.sin(2 * np.pi * t / 20)  # Low freq oscillation
    s3 = 1.5 * np.sin(2 * np.pi * t / 5)   # High freq oscillation (Target to extract)

    # 2. Generate a signal that ONLY exists in the reference, not the mixed channel
    s4 = 2.0 * np.sin(2 * np.pi * t / 11)  # Mid freq oscillation

    # 3. Create the weighted mixed channel and the reference channels
    # The mixed channel contains s1 and s2 with different weights, plus our target s3.
    # It does NOT contain s4.
    mixed = 0.5 * s1 + 2.0 * s2 + 1.0 * s3

    # Reference 1 contains s1, plus the distractor signal s4.
    ref1 = s1 + s4

    # Reference 2 is just s2.
    ref2 = s2

    # Create multivariate dataset
    X = np.column_stack((mixed, ref1, ref2))

    # 4. Fit MCissa
    L = 100
    print(f"Fitting MCissa with L={L}...")
    mcissa = MCissa(t=t, x=X)
    mcissa.fit(L=L)

    # Save standard component plot
    print("Saving standard components plot...")
    mcissa.plot_components(num_components=8)
    plt.savefig("mcissa_weighted_bss_components.png")
    plt.close()

    # 5. Extract the "leftover" signal (s3) from the mixed channel (Variable 0)
    print("Extracting target signal...")
    components_var0 = mcissa.Z_stacked[:, 0, :]

    # Find the components that match s3
    correlations = np.array([np.abs(np.corrcoef(components_var0[:, i], s3)[0, 1]) for i in range(components_var0.shape[1])])
    top_indices = np.argsort(correlations)[-2:]

    extracted_s3 = np.sum(components_var0[:N, top_indices], axis=1)

    # 6. Plot the results
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    axes[0].plot(mixed, label="Mixed Signal (0.5*s1 + 2.0*s2 + s3)", color="black")
    axes[0].set_title("Target Channel (Variable 0: Mixed)")
    axes[0].legend()

    axes[1].plot(s3, label="Ground Truth s3 (The Target Signal)", color="green", linestyle="--", linewidth=3)
    axes[1].plot(extracted_s3, label="Extracted s3 via M-CiSSA", color="blue", alpha=0.7)
    axes[1].set_title("Extraction of Target Signal from Weighted Mixed Channel")
    axes[1].legend()

    error = s3 - extracted_s3
    max_err = np.max(np.abs(error))
    print(f"Max extraction error for s3: {max_err}")

    axes[2].plot(error, label=f"Extraction Error (Max: {max_err:.2f})", color="red")
    axes[2].set_title("Extraction Error")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("mcissa_weighted_bss_extraction.png")
    print("Saved 'mcissa_weighted_bss_extraction.png'")

    print("Weighted BSS Example generation complete.")

if __name__ == "__main__":
    main()
