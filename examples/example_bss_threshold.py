import numpy as np
import matplotlib.pyplot as plt
from pycissa.processing.mcissa.mcissa import MCissa
import warnings
warnings.filterwarnings('ignore')

T = 300
t = np.arange(T)

# True signal (e.g. baseline physiological signal)
true_signal = 2.0 * np.sin(2 * np.pi * t / 80.0)

# Continuous Reference Source (e.g. continuous pressure or voltage)
continuous_ref = 3.0 * np.sin(2 * np.pi * t / 25.0)

# The Artifact ONLY occurs when the continuous reference exceeds a threshold
# and has a non-linear squared relationship to the reference when it does occur.
threshold = 1.5
artifact = np.zeros(T)
artifact[continuous_ref > threshold] = continuous_ref[continuous_ref > threshold]**2 * 0.5

# Main mixed signal
raw_mixed = true_signal + artifact + np.random.randn(T) * 0.2

# --- BAD APPROACH: Passing the raw continuous reference directly ---
# M-CiSSA is linear. If we give it the continuous reference, it will try to find a global
# linear fit for those frequencies and over-subtract them where the artifact DOES NOT exist.
X_bad = np.column_stack([raw_mixed, continuous_ref])
mcissa_bad = MCissa(t, X_bad)
# High alpha to skip surrogate test and just use variance threshold to pull the signal out
mcissa_bad.auto_blind_source_separation(L=60, main_index=0, K_surrogates=5, variance_threshold=0.01, alpha=1.0)
recovered_bad = mcissa_bad.x_cleaned

# --- GOOD APPROACH: Pre-thresholding the reference ---
# We derive a non-linear reference channel that matches the physical reality of the artifact.
derived_ref = np.zeros(T)
derived_ref[continuous_ref > threshold] = continuous_ref[continuous_ref > threshold]**2 * 0.5

X_good = np.column_stack([raw_mixed, derived_ref])
mcissa_good = MCissa(t, X_good)
# High alpha to skip surrogate test and just use variance threshold to pull the signal out
mcissa_good.auto_blind_source_separation(L=60, main_index=0, K_surrogates=5, variance_threshold=0.01, alpha=1.0)
recovered_good = mcissa_good.x_cleaned

# Calculate metrics
mse_raw = np.mean((raw_mixed - true_signal)**2)
mse_bad = np.mean((recovered_bad - true_signal)**2)
mse_good = np.mean((recovered_good - true_signal)**2)

print(f"MSE (Raw Mixed) : {mse_raw:.4f}")
print(f"MSE (Bad Linear Ref) : {mse_bad:.4f}")
print(f"MSE (Good Threshold Ref) : {mse_good:.4f}")

plt.figure(figsize=(12, 12))

plt.subplot(4, 1, 1)
plt.title(f"Original Components (MSE: {mse_raw:.2f})")
plt.plot(t, raw_mixed, label="Mixed Signal (True + Artifact)", color='lightgray')
plt.plot(t, true_signal, label="True Signal", color='black', linewidth=2)
plt.legend(loc='upper right')

plt.subplot(4, 1, 2)
plt.title("Reference Signals")
plt.plot(t, continuous_ref, label="Continuous Reference (Raw)", color='orange', alpha=0.7)
plt.plot(t, derived_ref, label=f"Derived Reference (Threshold > {threshold} squared)", color='red', linestyle='--')
plt.axhline(threshold, color='black', linestyle=':', label='Threshold')
plt.legend(loc='upper right')

plt.subplot(4, 1, 3)
plt.title(f"BAD APPROACH: Using Raw Continuous Reference (MSE: {mse_bad:.2f})")
plt.plot(t, true_signal, label="True Signal", color='black', linewidth=2)
plt.plot(t, recovered_bad, label="Recovered Signal (Linear Over-subtraction)", color='red', linestyle='--')
plt.legend(loc='upper right')

plt.subplot(4, 1, 4)
plt.title(f"GOOD APPROACH: Using Derived Thresholded Reference (MSE: {mse_good:.2f})")
plt.plot(t, true_signal, label="True Signal", color='black', linewidth=2)
plt.plot(t, recovered_good, label="Recovered Signal (Perfect Correction)", color='blue', linestyle='--')
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig("examples/bss_threshold_example_plot.png")
print("\nPlot saved as 'examples/bss_threshold_example_plot.png'")
