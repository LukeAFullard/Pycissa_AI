import numpy as np
import matplotlib.pyplot as plt
import os
from pycissa.processing.mcissa.mcissa import MCissa

output_dir = "examples/mcissa_benchmarks"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------
# 1. GENERATE THE DATASET (Simulating EEGdenoiseNet Model)
# ---------------------------------------------------------
np.random.seed(42)
sfreq = 250 # Sampling frequency in Hz
duration = 4 # Seconds
t = np.linspace(0, duration, sfreq * duration, endpoint=False)
n_samples = len(t)

# A. The Ground Truth: Clean EEG (Alpha & Beta waves + noise)
alpha_wave = 1.0 * np.sin(2 * np.pi * 10 * t) # 10 Hz Alpha
beta_wave = 0.5 * np.cos(2 * np.pi * 22 * t)  # 22 Hz Beta
clean_eeg = alpha_wave + beta_wave + np.random.normal(0, 0.2, n_samples)

# B. The Artifact: EOG (Eye Blinks)
# Eye blinks are low-frequency, high-amplitude spikes
eog_artifact = np.zeros(n_samples)
blink_times = [1.0, 2.5] # Blinks at 1s and 2.5s
for bt in blink_times:
    idx = int(bt * sfreq)
    # Create a rough Gaussian spike shape for the blink
    width = 30
    window = np.exp(-0.5 * ((np.arange(-width, width) / (width/3))**2))
    eog_artifact[idx-width:idx+width] += window * 8.0 # Very high amplitude

eog_artifact += np.random.normal(0, 0.1, n_samples) # Add some measurement noise to EOG

# C. The Contaminated Signal (x = x_clean + lambda * artifact)
lambda_factor = 1.2 # Severe contamination
contaminated_eeg = clean_eeg + (lambda_factor * eog_artifact)

# ---------------------------------------------------------
# 2. RUN M-CISSA BLIND SOURCE SEPARATION
# ---------------------------------------------------------
print("Running M-CiSSA Blind Source Separation...")

# Create the multivariate dataset: [Contaminated EEG, Reference EOG]
X = np.column_stack((contaminated_eeg, eog_artifact))

# Initialize MCissa
mcissa = MCissa(t=t, x=X)
# Window length: Needs to be long enough to capture the low-freq blink shape
mcissa.fit(L=int(sfreq * 0.5))

# Run BSS: Clean channel 0 using channel 1 as reference.
# We set alpha=1.0 to guarantee separation purely on variance for this clean test.
mcissa.auto_blind_source_separation(main_index=0, reference_indices=[1], alpha=1.0)

cleaned_eeg = mcissa.x_cleaned

# ---------------------------------------------------------
# 3. CALCULATE VERIFICATION METRICS
# ---------------------------------------------------------
# Metric 1: Correlation Coefficient (CC) - Target: >= 0.85
cc = np.corrcoef(clean_eeg, cleaned_eeg)[0, 1]

# Metric 2: Relative Root Mean Squared Error (RRMSE) - Target: <= 0.45
rmse = np.sqrt(np.mean((clean_eeg - cleaned_eeg)**2))
rms_clean = np.sqrt(np.mean(clean_eeg**2))
rrmse = rmse / rms_clean

# Metric 3: Instantaneous Error
instantaneous_error = clean_eeg - cleaned_eeg

print(f"\n--- VERIFICATION RESULTS ---")
print(f"Target Correlation Coefficient (CC) >= 0.85")
print(f"Actual CC: {cc:.4f} {'(PASS)' if cc >= 0.85 else '(FAIL)'}")

print(f"\nTarget Relative RMSE (RRMSE) <= 0.45")
print(f"Actual RRMSE: {rrmse:.4f} {'(PASS)' if rrmse <= 0.45 else '(FAIL)'}")

# ---------------------------------------------------------
# 4. PLOT THE RESULTS
# ---------------------------------------------------------
fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)

axes[0].plot(t, clean_eeg, color='green')
axes[0].set_title("1. Ground Truth (Clean EEG - Target to Recover)")
axes[0].set_ylim(-12, 12)

axes[1].plot(t, eog_artifact, color='red')
axes[1].set_title("2. EOG Reference Channel (Eye Blinks)")
axes[1].set_ylim(-12, 12)

axes[2].plot(t, contaminated_eeg, color='grey')
axes[2].set_title("3. Contaminated EEG (Input to M-CiSSA)")
axes[2].set_ylim(-12, 12)

axes[3].plot(t, cleaned_eeg, color='blue')
axes[3].set_title(f"4. M-CiSSA Cleaned Output (CC: {cc:.2f})")
axes[3].set_ylim(-12, 12)

axes[4].plot(t, instantaneous_error, color='purple')
axes[4].set_title("5. Instantaneous Error (Ground Truth - M-CiSSA Output)")
axes[4].set_ylim(-12, 12)
axes[4].axhline(0, color='black', linestyle='--', linewidth=0.5)

plt.xlabel("Time (Seconds)")
plt.tight_layout()

save_path = os.path.join(output_dir, "eeg_bss_results.png")
plt.savefig(save_path, bbox_inches='tight')
plt.close()

print(f"\nSaved visualization to: {save_path}")
