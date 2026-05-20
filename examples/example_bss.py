import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from pycissa.processing.mcissa.mcissa import MCissa

T = 300
t = np.arange(T)

# The "True" signal is a slow cycle
true_signal = 5.0 * np.sin(2 * np.pi * t / 50.0)

# Random noise processes
ref1 = np.random.randn(T) * 1.0
ref2 = np.random.randn(T) * 1.0

# Add a massive artifact!
artifact = 50.0 * np.sin(2 * np.pi * t / 12.0)

# Mix heavily into the main signal!
main_signal = true_signal + 10.0 * ref1 + 5.0 * ref2 + artifact
ref_channel_1 = ref1 + artifact

X = np.column_stack([main_signal, ref_channel_1, ref2])
mcissa = MCissa(t, X)

mcissa.auto_blind_source_separation(L=60, main_index=0, K_surrogates=5, variance_threshold=0.01)

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
plt.plot(t, ref_channel_1, label="Reference 1", color='orange')
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
