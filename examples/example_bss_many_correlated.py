import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from pycissa.processing.mcissa.mcissa import MCissa

# Reduced T and L to make the Monte Carlo test fast enough for the example
T = 300
t = np.arange(T)

# True main signal
true_signal = 3.0 * np.sin(2 * np.pi * t / 60.0) + 1.5 * np.sin(2 * np.pi * t / 15.0)

# Many correlated artifacts (e.g. 4 distinct muscle/eye movement sensors)
art_src_1 = 5.0 * np.sin(2 * np.pi * t / 45.0)  # Drift
art_src_2 = 2.0 * np.sin(2 * np.pi * t / 6.0)   # Hum
art_src_3 = 3.0 * np.sin(2 * np.pi * t / 25.0)  # Muscle artifact

# Mix these sources into 4 reference channels with different weights and phase shifts
ref_1 = 1.0 * art_src_1 + 0.5 * art_src_2 + np.random.randn(T) * 0.2
ref_2 = 0.8 * art_src_1 + 1.2 * art_src_3 + np.random.randn(T) * 0.2
ref_3 = 0.2 * art_src_2 + 0.9 * art_src_3 + np.random.randn(T) * 0.2
# ref_4 has a delayed version of art_src_1
ref_4 = 1.5 * (5.0 * np.sin(2 * np.pi * (t-5) / 45.0)) + np.random.randn(T) * 0.2

# The main signal is contaminated by a complex mix of these sources
main_contamination = (
    1.2 * art_src_1 +
    0.8 * art_src_2 +
    1.5 * art_src_3 +
    0.5 * ref_4  # Add the delayed source too
)

raw_mixed = true_signal + main_contamination + np.random.randn(T) * 0.5

X = np.column_stack([raw_mixed, ref_1, ref_2, ref_3, ref_4])

# We use Auto BSS on the 5-channel matrix
mcissa = MCissa(t, X)

# Using actual Monte Carlo test to correctly identify significance in references.
# K_surrogates=10 is used here to keep execution time short for the example.
mcissa.auto_blind_source_separation(
    L=60,
    main_index=0,
    K_surrogates=10,
    variance_threshold=0.01,
    alpha=0.05,
    trend_always_significant=False
)

recovered_signal = mcissa.x_cleaned

mse_raw = np.mean((raw_mixed - true_signal)**2)
mse_cleaned = np.mean((recovered_signal - true_signal)**2)

print(f"MSE (Raw Mixed vs True) : {mse_raw:.4f}")
print(f"MSE (Recovered vs True) : {mse_cleaned:.4f}")

plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.title(f"Original Components (MSE: {mse_raw:.2f})")
plt.plot(t, raw_mixed, label="Heavily Contaminated Signal", color='lightgray')
plt.plot(t, true_signal, label="True Signal", color='black', linewidth=2)
plt.legend(loc='upper right')

plt.subplot(3, 1, 2)
plt.title("4 Correlated Reference Channels")
plt.plot(t, ref_1, label="Ref 1 (Drift + Hum)", alpha=0.7)
plt.plot(t, ref_2, label="Ref 2 (Drift + Muscle)", alpha=0.7)
plt.plot(t, ref_3, label="Ref 3 (Hum + Muscle)", alpha=0.7)
plt.plot(t, ref_4, label="Ref 4 (Delayed Drift)", alpha=0.7)
plt.legend(loc='upper right', fontsize='small')

plt.subplot(3, 1, 3)
plt.title(f"M-CiSSA Recovery with Multiple References (MSE: {mse_cleaned:.2f})")
plt.plot(t, true_signal, label="True Signal", color='black', linewidth=2)
plt.plot(t, recovered_signal, label="Recovered Signal", color='blue', linestyle='--')
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig("examples/bss_many_correlated_example_plot.png")
print("\nPlot saved as 'examples/bss_many_correlated_example_plot.png'")
