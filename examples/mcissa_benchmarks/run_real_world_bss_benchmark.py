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
df = sm.datasets.sunspots.load_pandas().data
raw_sunspots = df['SUNACTIVITY'].values
t_years = df['YEAR'].values
n_samples = len(t_years)

# ---------------------------------------------------------
# 2. CREATE A CONTROLLED REAL-WORLD BSS SCENARIO
# ---------------------------------------------------------
np.random.seed(42)
# Create a strong 5-year sensor drift/interference
interference_signal = 40.0 * np.sin(2 * np.pi * t_years / 5.0)

# The "Mixed" Main Channel
contaminated_sunspots = raw_sunspots + interference_signal

# The Reference Channel
reference_sensor = interference_signal + np.random.normal(0, 5.0, n_samples)

X = np.column_stack((contaminated_sunspots, reference_sensor))

# ---------------------------------------------------------
# 3. RUN M-CISSA BLIND SOURCE SEPARATION
# ---------------------------------------------------------
print("Running M-CiSSA Blind Source Separation...")
mcissa = MCissa(t=t_years, x=X)
mcissa.fit(L=22)
mcissa.auto_blind_source_separation(main_index=0, reference_indices=[1], alpha=1.0)

extracted_sunspots = mcissa.x_cleaned

# ---------------------------------------------------------
# 4. VERIFY AGAINST THE REAL-WORLD GROUND TRUTH
# ---------------------------------------------------------
cc = np.corrcoef(raw_sunspots, extracted_sunspots)[0, 1]

# Calculate instantaneous error
instantaneous_error = raw_sunspots - extracted_sunspots

print(f"\n--- VERIFICATION RESULTS ---")
print(f"Task: Extract Real Sunspot Activity from Corrupted Sensor Data.")
print(f"Target Correlation: > 0.95")
print(f"M-CiSSA Extraction Correlation: {cc:.4f} {'(PASS)' if cc >= 0.95 else '(FAIL)'}")

# ---------------------------------------------------------
# 5. PLOT THE RESULTS
# ---------------------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

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

# Plot 4: Instantaneous Error
axes[3].plot(t_years, instantaneous_error, color='purple', label='Error')
axes[3].set_title("4. Instantaneous Error (Ground Truth - M-CiSSA Output)")
axes[3].axhline(0, color='black', linestyle='--', linewidth=0.5)
# Set y-limits roughly similar to the signal scale for perspective
axes[3].set_ylim(-60, 60)
axes[3].legend(loc='upper right')

plt.xlabel("Year")
plt.tight_layout()

save_path = os.path.join(output_dir, "real_world_bss_results.png")
plt.savefig(save_path, bbox_inches='tight')
plt.close()

print(f"\nSaved visualization to: {save_path}")
