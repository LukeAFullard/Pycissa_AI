import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from pycissa.processing.mcissa.mcissa import MCissa

T = 300
t = np.arange(T)

# The "True" signal is a slow cycle AND a shared frequency!
# Note that we put the T=12 component in BOTH the true signal and the artifact,
# just to show that MCissa spatial eigenvectors can separate them!
true_signal = 5.0 * np.sin(2 * np.pi * t / 50.0) + 2.0 * np.sin(2 * np.pi * t / 12.0)

# Artifacts
ref1 = 20.0 * np.sin(2 * np.pi * t / 12.0 + np.pi/4) # Same freq as signal part, but shifted/independent spatial source
ref2 = 10.0 * np.sin(2 * np.pi * t / 5.0)

# Contaminate main signal
main_signal = true_signal + 1.0 * ref1 + 1.0 * ref2 + np.random.randn(T) * 1.0

# Mix into references
ref_channel_1 = ref1 + np.random.randn(T) * 0.5
ref_channel_2 = ref2 + np.random.randn(T) * 0.5

X = np.column_stack([main_signal, ref_channel_1, ref_channel_2])
mcissa = MCissa(t, X)

# Using auto BSS based on Subcomponent variance checking!
mcissa.auto_blind_source_separation(L=60, main_index=0, K_surrogates=5, variance_threshold=0.10)

mse_original_to_true = np.mean((main_signal - true_signal)**2)
mse_cleaned_to_true = np.mean((mcissa.x_cleaned - true_signal)**2)

print(f"\nMSE (Raw Mixed Signal vs True Signal): {mse_original_to_true:.4f}")
print(f"MSE (Cleaned Signal vs True Signal)  : {mse_cleaned_to_true:.4f}")

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.title(f"Original Contaminated Signal (MSE: {mse_original_to_true:.2f})")
plt.plot(t, main_signal, label="Raw Contaminated Signal", color='lightgray')
plt.plot(t, true_signal, label="True Signal", color='black', linewidth=2)
plt.legend()

plt.subplot(3, 1, 2)
plt.title("Reference Channels (Artifacts)")
plt.plot(t, ref1, label="Reference 1", color='orange')
plt.plot(t, ref2, label="Reference 2", color='red')
plt.legend()

plt.subplot(3, 1, 3)
plt.title(f"M-CiSSA BSS Recovered Signal (MSE: {mse_cleaned_to_true:.2f})")
plt.plot(t, true_signal, label="True Signal", color='black', linewidth=2)
plt.plot(t, mcissa.x_cleaned, label="Recovered (Cleaned) Signal", color='blue', linestyle='--')
plt.legend()

plt.tight_layout()
plt.savefig("examples/bss_example_plot.png")
print("\nPlot saved as 'examples/bss_example_plot.png'")
