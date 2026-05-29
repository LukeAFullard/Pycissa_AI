import numpy as np
import statsmodels.api as sm
from pycissa.processing.mcissa.mcissa import MCissa
import matplotlib.pyplot as plt
import os

output_dir = "examples/mcissa_benchmarks"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------
# 1. DOWNLOAD REAL-WORLD DATA
# ---------------------------------------------------------
print("Downloading Real-World Dataset (Sunspots)...")
# We use the classic Sunspots dataset (Yearly data from 1700 to 2008).
# This is a widely used time-series benchmark due to its well-known ~11-year solar cycle.
df = sm.datasets.sunspots.load_pandas().data
raw_sunspots = df['SUNACTIVITY'].values
t_years = df['YEAR'].values
n_samples = len(t_years)

# ---------------------------------------------------------
# 2. CREATE A CONTROLLED REAL-WORLD BSS SCENARIO
# ---------------------------------------------------------
# In many BSS tasks (like satellite imagery or telescope sensor data),
# the real-world signal of interest (sunspots) is corrupted by a known periodic interference
# (e.g., orbital mechanics, sensor calibration drift, or climate oscillations).

np.random.seed(42)
# Create a strong 5-year sensor drift/interference
interference_signal = 40.0 * np.sin(2 * np.pi * t_years / 5.0)

# The "Mixed" Main Channel: The real sunspot data heavily corrupted by the interference
contaminated_sunspots = raw_sunspots + interference_signal

# The Reference Channel: A measurement of the interference (with some ambient noise)
reference_sensor = interference_signal + np.random.normal(0, 5.0, n_samples)

X = np.column_stack((contaminated_sunspots, reference_sensor))

# ---------------------------------------------------------
# 3. RUN M-CISSA BLIND SOURCE SEPARATION
# ---------------------------------------------------------
print("Running M-CiSSA Blind Source Separation...")
mcissa = MCissa(t=t_years, x=X)

# We use an L of 22 years (to comfortably capture both the 11-year solar cycle and 5-year interference)
mcissa.fit(L=22)

# Run BSS: Clean the contaminated sunspots (0) using the reference sensor (1)
mcissa.auto_blind_source_separation(main_index=0, reference_indices=[1], alpha=1.0)

extracted_sunspots = mcissa.x_cleaned

# ---------------------------------------------------------
# 4. VERIFY AGAINST THE REAL-WORLD GROUND TRUTH
# ---------------------------------------------------------
# Because we started with the raw, real-world sunspot data before corrupting it,
# we possess the exact mathematical "Ground Truth" for this specific real-world dataset.
cc = np.corrcoef(raw_sunspots, extracted_sunspots)[0, 1]

print(f"\n--- VERIFICATION RESULTS ---")
print(f"Task: Extract Real Sunspot Activity from Corrupted Sensor Data.")
print(f"Target Correlation: > 0.95")
print(f"M-CiSSA Extraction Correlation: {cc:.4f} {'(PASS)' if cc >= 0.95 else '(FAIL)'}")

# ---------------------------------------------------------
# 5. PLOT THE RESULTS
# ---------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Plot 1: Ground Truth vs Contaminated
axes[0].plot(t_years, raw_sunspots, color='green', label='Ground Truth (Real Sunspots)')
axes[0].plot(t_years, contaminated_sunspots, color='grey', alpha=0.7, label='Contaminated Signal (Input)')
axes[0].set_title("1. The Real-World Target vs The Contaminated Input")
axes[0].legend(loc='upper right')

# Plot 2: The Reference Channel
axes[1].plot(t_years, reference_sensor, color='red', label='Reference Sensor (Interference)')
axes[1].set_title("2. The Reference Channel")
axes[1].legend(loc='upper right')

# Plot 3: The Cleaned Output
axes[2].plot(t_years, raw_sunspots, color='green', linestyle='--', alpha=0.5, label='Ground Truth')
axes[2].plot(t_years, extracted_sunspots, color='blue', label=f'M-CiSSA Extracted (CC: {cc:.4f})')
axes[2].set_title("3. BSS Extraction vs Ground Truth")
axes[2].legend(loc='upper right')

plt.xlabel("Year")
plt.tight_layout()

save_path = os.path.join(output_dir, "real_world_bss_results.png")
plt.savefig(save_path, bbox_inches='tight')
plt.close()

print(f"\nSaved visualization to: {save_path}")
