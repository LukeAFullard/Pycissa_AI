import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from pycissa.processing.mcissa.mcissa import MCissa

np.random.seed(42)
T = 300
t = np.arange(1, T + 1)
true_main_signal = 2.0 * np.sin(2 * np.pi * t / 15)

# Interference source
interference = 5.0 * np.sin(2 * np.pi * t / 6)

# Main channel contaminated with interference
main_mixed = true_main_signal + interference + np.random.normal(0, 0.2, T)

# Two reference signals that are related (both measure the interference)
ref1 = interference + np.random.normal(0, 0.5, T)
ref2 = 0.8 * interference + np.random.normal(0, 0.5, T) # Highly related to ref1

X = np.column_stack((main_mixed, ref1, ref2))

print("Running M-CISSA Blind Source Separation with Related References...")
mcissa = MCissa(t, X)
mcissa.fit(L=48)
# We set alpha=1.0 to bypass the strict MC surrogate significance check for deterministic variance separation in this example.
mcissa.auto_blind_source_separation(main_index=0, reference_indices=[1, 2], alpha=1.0)

clean_corr = np.corrcoef(true_main_signal, mcissa.x_cleaned)[0, 1]
print(f"Clean Signal Correlation with True Main: {clean_corr:.4f}")

plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.plot(t, true_main_signal, label="True Main Signal", linewidth=2)
plt.plot(t, main_mixed, label="Mixed Channel 0 (Main)", alpha=0.5)
plt.title("Main Signal Before BSS")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, ref1, label="Reference 1 (Interference)", alpha=0.7)
plt.plot(t, ref2, label="Reference 2 (Related Interference)", alpha=0.7)
plt.title("Reference Channels (Highly Related)")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, true_main_signal, label="True Main Signal", linestyle='dashed', linewidth=2)
plt.plot(t, mcissa.x_cleaned, label="Extracted Cleaned Signal", linewidth=2)
plt.title(f"Cleaned Signal After BSS (Corr: {clean_corr:.3f})")
plt.legend()

plt.tight_layout()
plt.savefig("examples/mcissa/bss_related_references.png")
print("Saved plot to examples/mcissa/bss_related_references.png")
