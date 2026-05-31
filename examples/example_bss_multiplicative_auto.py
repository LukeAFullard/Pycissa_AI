import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from pycissa.processing.mcissa.mcissa import MCissa
from pycissa.preprocessing import test_if_multiplicative

T = 400
t = np.arange(T)

# True main signal (e.g. underlying physiological cycle)
true_signal = 5.0 * np.sin(2 * np.pi * t / 25.0)

# The Artifact modulates the AMPLITUDE (variance) of the true signal
# e.g. A movement artifact that scales the gain of the sensor
artifact_modulation = 1.0 + 0.8 * np.sin(2 * np.pi * t / 80.0)

# True main signal mixed multiplicatively
raw_mixed = true_signal * artifact_modulation + np.random.randn(T) * 0.5

# Reference channel for the artifact
ref_channel = artifact_modulation + np.random.randn(T) * 0.2

# 1. Variance Correlation Test
# This automatically checks if the artifact correlates with the rolling standard deviation
# rather than just the raw mean, indicating a multiplicative relationship.
is_mult, corr_raw, corr_std = test_if_multiplicative(raw_mixed, ref_channel, window_size=20)

print(f"Variance Correlation Test:")
print(f"Is Multiplicative? {is_mult}")
print(f"Raw Correlation (Mean): {corr_raw:.2f}")
print(f"Variance Correlation (Std): {corr_std:.2f}")

# 2. Linear BSS on Multiplicative Data
# Because the artifact modulates the amplitude, the frequencies of the true signal
# and the artifact mix (cross-modulation), creating sidebands. Linear BSS struggles
# to perfectly separate this because the artifact doesn't exist as a simple additive wave.
X = np.column_stack([raw_mixed, ref_channel])
mcissa = MCissa(t, X)

# We use variance_threshold with alpha=1.0 for a fast linear extraction demonstration
mcissa.auto_blind_source_separation(L=60, main_index=0, K_surrogates=5, variance_threshold=0.01, alpha=1.0)
recovered_linear = mcissa.x_cleaned

mse_mixed = np.mean((raw_mixed - true_signal)**2)
mse_linear = np.mean((recovered_linear - true_signal)**2)

print(f"\nMSE (Raw Mixed) : {mse_mixed:.4f}")
print(f"MSE (Linear BSS): {mse_linear:.4f}")


plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.title(f"Original Components (MSE: {mse_mixed:.2f})")
plt.plot(t, raw_mixed, label="Multiplicatively Mixed", color='lightgray')
plt.plot(t, true_signal, label="True Signal", color='black', linewidth=2)
plt.legend(loc="upper right")

plt.subplot(3, 1, 2)
plt.title("Reference Signal vs Mixed Envelope")
import pandas as pd
rolling_std = pd.Series(raw_mixed).rolling(window=20, center=True).std().bfill().ffill().values
plt.plot(t, rolling_std, label=f"Mixed Envelope (Variance Corr: {corr_std:.2f})", color='purple', linestyle='--')
plt.plot(t, ref_channel, label="Artifact Reference Channel", color='orange', alpha=0.7)
plt.legend(loc="upper right")

plt.subplot(3, 1, 3)
plt.title(f"Linear M-CiSSA Recovery on Multiplicative Data (MSE: {mse_linear:.2f})")
plt.plot(t, true_signal, label="True Signal", color='black', linewidth=2)
plt.plot(t, recovered_linear, label="Recovered (Linear)", color='red', linestyle='--')
plt.text(50, 4, "Notice the persistent amplitude wobble due to cross-modulation sidebands", color='red')
plt.legend(loc="upper right")

plt.tight_layout()
plt.savefig("examples/bss_multiplicative_auto_test.png")
print("\nPlot saved as 'examples/bss_multiplicative_auto_test.png'")
