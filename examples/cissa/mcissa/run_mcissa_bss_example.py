import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure pycissa is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from pycissa.processing.mcissa.mcissa import MCissa

def main():
    print("Generating M-CiSSA Blind Source Separation example...")
    np.random.seed(42)
    N = 300
    t = np.arange(N)

    # 1. Generate three independent pure signals
    s1 = np.linspace(0, 10, N)  # Trend
    s2 = 3.0 * np.sin(2 * np.pi * t / 20)  # Low freq oscillation
    s3 = 1.5 * np.sin(2 * np.pi * t / 5)   # High freq oscillation

    # 2. Mix them together (this is our target channel)
    mixed = s1 + s2 + s3

    # 3. Create a multivariate dataset containing the mixed signal (Var 0)
    # and two of the pure signals as "reference" channels (Var 1 and Var 2)
    X = np.column_stack((mixed, s1, s2))

    # 4. Fit MCissa
    L = 100
    print(f"Fitting MCissa with L={L}...")
    mcissa = MCissa(t=t, x=X)
    mcissa.fit(L=L)

    # Save standard component plot
    print("Saving standard components plot...")
    mcissa.plot_components(num_components=6)
    plt.savefig("mcissa_bss_components.png")
    plt.close()

    # 5. Extract the "leftover" signal (s3) from the mixed channel (Variable 0)
    # The reconstructed components for variable 0 are in mcissa.Z_stacked[:, 0, :]
    print("Extracting target signal...")
    components_var0 = mcissa.Z_stacked[:, 0, :]

    # To isolate s3, we find the components that represent it.
    # Because M-CiSSA jointly diagonalizes the dataset, the variance associated with s1 and s2
    # gets strongly pulled into components that correlate with the reference channels.
    # The leftover variance in the mixed channel naturally isolates s3.
    # Here we just use simple correlation to grab the highest matching components for plotting,
    # though in a real blind scenario, you would look at the eigenvalue groupings or frequencies.
    correlations = np.array([np.abs(np.corrcoef(components_var0[:, i], s3)[0, 1]) for i in range(components_var0.shape[1])])

    # Get the indices of the top 2 components (since an oscillation splits into 2 components)
    top_indices = np.argsort(correlations)[-2:]

    # Reconstruct s3 from these two components (truncate to N length)
    extracted_s3 = np.sum(components_var0[:N, top_indices], axis=1)

    # 6. Plot the results
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    axes[0].plot(mixed, label="Mixed Signal (s1 + s2 + s3)", color="black")
    axes[0].set_title("Target Channel (Variable 0: Mixed)")
    axes[0].legend()

    axes[1].plot(s3, label="Ground Truth s3 (The 'Leftover' Signal)", color="green", linestyle="--", linewidth=3)
    axes[1].plot(extracted_s3, label="Extracted s3 via M-CiSSA", color="blue", alpha=0.7)
    axes[1].set_title("Extraction of Leftover Signal from Mixed Channel")
    axes[1].legend()

    error = s3 - extracted_s3
    max_err = np.max(np.abs(error))
    print(f"Max extraction error for s3: {max_err}")

    axes[2].plot(error, label=f"Extraction Error (Max: {max_err:.2f})", color="red")
    axes[2].set_title("Extraction Error")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("mcissa_bss_extraction.png")
    print("Saved 'mcissa_bss_extraction.png'")

    print("BSS Example generation complete.")

if __name__ == "__main__":
    main()
