import numpy as np
import matplotlib.pyplot as plt
import os
from pycissa.processing.mcissa.mcissa import MCissa

output_dir = "examples/mcissa_benchmarks"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------
# 1. GENERATE THE DATASET (Simulating Cocktail Party Model)
# ---------------------------------------------------------
np.random.seed(0)
sfreq = 500 # Simulated audio sampling frequency
duration = 2 # Seconds
t = np.linspace(0, duration, sfreq * duration, endpoint=False)

# A. Ground Truth Source 1: "Vocals" (High Frequency, Modulated)
# Simulating a singer holding a vibrato note
source_1 = np.sin(2 * np.pi * 30 * t) * np.sin(2 * np.pi * 2 * t)

# B. Ground Truth Source 2: "Bass" (Low Frequency, Steady)
source_2 = 1.5 * np.sign(np.sin(2 * np.pi * 5 * t)) # Square wave to simulate a punchy bassline

S = np.c_[source_1, source_2]
S += 0.1 * np.random.normal(size=S.shape) # Add ambient room noise

# C. The Mixing Process (The Microphones)
# Microphone 1 is closer to the singer. Microphone 2 is closer to the bass amp.
mixing_matrix = np.array([
    [1.0, 0.4],  # Mic 1: 100% Vocals, 40% Bass
    [0.3, 1.0]   # Mic 2: 30% Vocals, 100% Bass
])

# X contains the two mixed audio channels
X = np.dot(S, mixing_matrix.T)

# ---------------------------------------------------------
# 2. RUN M-CISSA BLIND SOURCE SEPARATION
# ---------------------------------------------------------
print("Running M-CiSSA Cocktail Party Separation...")

mcissa = MCissa(t=t, x=X)
mcissa.fit(L=100) # Window length must be long enough to capture the 5Hz bass wave

# We want to extract Source 1 (Vocals) from Mic 1.
# We use Mic 2 (which is dominated by the Bass) as the reference to cancel out the Bass interference.
mcissa.auto_blind_source_separation(main_index=0, reference_indices=[1], alpha=1.0, trend_always_significant=False)

extracted_vocals = mcissa.x_cleaned
extracted_bass_interference = mcissa.x_influence # The bass that was removed

# ---------------------------------------------------------
# 3. CALCULATE VERIFICATION METRICS
# ---------------------------------------------------------
# Target: Correlation > 0.90
cc_vocals = np.corrcoef(source_1, extracted_vocals)[0, 1]
cc_bass = np.corrcoef(source_2, extracted_bass_interference)[0, 1]

print(f"\n--- VERIFICATION RESULTS ---")
print(f"Target Correlation Coefficient (CC) >= 0.90")
print(f"Vocals Extraction CC: {cc_vocals:.4f} {'(PASS)' if cc_vocals >= 0.90 else '(FAIL)'}")
print(f"Bass Interference Isolation CC: {cc_bass:.4f} {'(PASS)' if cc_bass >= 0.90 else '(FAIL)'}")

# ---------------------------------------------------------
# 4. PLOT THE RESULTS
# ---------------------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

axes[0].plot(t, source_1, color='green', label='Source 1 (Vocals)')
axes[0].plot(t, source_2, color='darkred', alpha=0.6, label='Source 2 (Bass)')
axes[0].set_title("1. The Ground Truth Unmixed Sources")
axes[0].legend(loc='upper right')

axes[1].plot(t, X[:, 0], color='grey', label='Mic 1')
axes[1].plot(t, X[:, 1], color='black', alpha=0.6, label='Mic 2')
axes[1].set_title("2. The Mixed Microphones (Cocktail Party Problem)")
axes[1].legend(loc='upper right')

axes[2].plot(t, extracted_vocals, color='blue', label='Extracted Vocals')
axes[2].set_title(f"3. M-CiSSA Cleaned Vocals (CC against Source 1: {cc_vocals:.2f})")
axes[2].legend(loc='upper right')

# Calculate the instantaneous error for the vocals
error = source_1 - extracted_vocals
axes[3].plot(t, error, color='purple', label='Error')
axes[3].set_title("4. Instantaneous Error (Ground Truth Vocals - Extracted Vocals)")
axes[3].axhline(0, color='black', linestyle='--', linewidth=0.5)
axes[3].set_ylim(-2, 2)
axes[3].legend(loc='upper right')

plt.xlabel("Time (Seconds)")
plt.tight_layout()

save_path = os.path.join(output_dir, "audio_bss_results.png")
plt.savefig(save_path, bbox_inches='tight')
plt.close()

print(f"\nSaved visualization to: {save_path}")
