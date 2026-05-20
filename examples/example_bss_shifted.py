import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from pycissa.processing.mcissa.mcissa import MCissa

T = 300
t = np.arange(T)

# The "True" signal is a slow cycle AND a shared frequency!
true_signal = 5.0 * np.sin(2 * np.pi * t / 50.0) + 2.0 * np.sin(2 * np.pi * t / 12.0)

# Artifact (Climate Change)
# Let's say climate change operates on a 12-month cycle
climate_change = 20.0 * np.sin(2 * np.pi * t / 12.0)

# The river flow is impacted by climate change, but delayed by 3 months! (Phase shift of 3/12 * 2pi = pi/2)
climate_impact_on_river = 20.0 * np.sin(2 * np.pi * (t - 3) / 12.0)

# Contaminate main signal (River Flow)
river_flow = true_signal + climate_impact_on_river + np.random.randn(T) * 1.0

# Mix into references (Climate Change reading)
ref_climate = climate_change + np.random.randn(T) * 0.5

X = np.column_stack([river_flow, ref_climate])
mcissa = MCissa(t, X)

# Using auto BSS! M-CiSSA natively handles phase shifts using its sine/cosine pairs.
mcissa.auto_blind_source_separation(L=60, main_index=0, K_surrogates=5, variance_threshold=0.10)

mse_original_to_true = np.mean((river_flow - true_signal)**2)
mse_cleaned_to_true = np.mean((mcissa.x_cleaned - true_signal)**2)

print(f"\nMSE (Raw River Flow vs True River Flow): {mse_original_to_true:.4f}")
print(f"MSE (Cleaned River Flow vs True River Flow)  : {mse_cleaned_to_true:.4f}")

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.title(f"Original River Flow (MSE: {mse_original_to_true:.2f})")
plt.plot(t, river_flow, label="Raw River Flow (with delayed Climate impact)", color='lightgray')
plt.plot(t, true_signal, label="True River Flow", color='black', linewidth=2)
plt.legend()

plt.subplot(3, 1, 2)
plt.title("Reference Channel")
plt.plot(t, ref_climate, label="Climate Change (0 delay)", color='orange')
plt.plot(t, climate_impact_on_river, label="Climate Impact on River (3 month delay)", color='red', linestyle='--')
plt.legend()

plt.subplot(3, 1, 3)
plt.title(f"M-CiSSA BSS Recovered River Flow (MSE: {mse_cleaned_to_true:.2f})")
plt.plot(t, true_signal, label="True River Flow", color='black', linewidth=2)
plt.plot(t, mcissa.x_cleaned, label="Recovered River Flow", color='blue', linestyle='--')
plt.legend()

plt.tight_layout()
plt.savefig("examples/bss_shifted_example_plot.png")
print("\nPlot saved as 'examples/bss_shifted_example_plot.png'")
