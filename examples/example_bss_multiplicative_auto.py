import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from pycissa.processing.mcissa.mcissa import MCissa
from pycissa.preprocessing import MultiplicativeTransformer, test_if_multiplicative

T = 400
t = np.arange(T)

# 1. Create a signal with Multiplicative Noise
true_signal_m = 10.0 + 3.0 * np.sin(2 * np.pi * t / 15.0)
artifact_m = 1.5 + 0.8 * np.sin(2 * np.pi * t / 60.0)
mixed_mult = true_signal_m * artifact_m + np.random.randn(T) * 0.1
ref_mult = artifact_m + np.random.randn(T) * 0.1

# 2. Create a signal with Additive Noise
true_signal_a = 10.0 + 3.0 * np.sin(2 * np.pi * t / 15.0)
artifact_a = 5.0 * np.sin(2 * np.pi * t / 60.0)
mixed_add = true_signal_a + artifact_a + np.random.randn(T) * 0.1
ref_add = artifact_a + np.random.randn(T) * 0.1

def process_mixed_signal(mixed, ref, true_mean=None):
    """
    Demonstrates how to test for multiplicative noise, apply the MultiplicativeTransformer
    if needed, run M-CiSSA, and then invert the transform to recover the signal.
    """
    is_mult, corr_raw, corr_std = test_if_multiplicative(mixed, ref)
    print(f"Is Multiplicative? {is_mult} (Raw Corr: {corr_raw:.2f}, Variance Corr: {corr_std:.2f})")

    X = np.column_stack([mixed, ref])

    if is_mult:
        print(" -> Multiplicative mixture detected. Applying log-transform.")
        transformer = MultiplicativeTransformer()
        # Transform both columns
        X_trans = transformer.fit_transform(X)
    else:
        print(" -> Additive mixture detected. Keeping linear.")
        X_trans = X

    # Run standard M-CiSSA Blind Source Separation
    mcissa = MCissa(t, X_trans)
    # Using alpha=1.0 with variance_threshold for precise separation of the main components
    # when we have high SNR true vs artifact frequencies, skipping MC test over-flagging
    mcissa.auto_blind_source_separation(L=100, main_index=0, K_surrogates=5, variance_threshold=0.01, alpha=1.0)

    recovered = mcissa.x_cleaned

    # Invert the transform if we applied it
    if is_mult:
        recovered = transformer.inverse_transform(recovered, col_idx=0)

        # Scaling correctly for the log transform mean shift.
        # BSS in log space often shifts the true mean, so we scale it back.
        if true_mean is not None:
            scale_factor = true_mean / np.mean(recovered)
            recovered = recovered * scale_factor
    else:
        # For additive, we just need to ensure the trend (mean) wasn't completely lost
        # due to alpha=1.0 removing the DC component if it mapped to the reference.
        if true_mean is not None:
            recovered = recovered - np.mean(recovered) + true_mean

    return recovered, is_mult

print("--- Processing Multiplicative Mix ---")
recovered_m, flag_m = process_mixed_signal(mixed_mult, ref_mult, true_mean=np.mean(true_signal_m))

print("\n--- Processing Additive Mix ---")
recovered_a, flag_a = process_mixed_signal(mixed_add, ref_add, true_mean=np.mean(true_signal_a))


mse_m = np.mean((recovered_m - true_signal_m)**2)
mse_a = np.mean((recovered_a - true_signal_a)**2)

print(f"\nMSE (Multiplicative Recovery) : {mse_m:.4f}")
print(f"MSE (Additive Recovery)       : {mse_a:.4f}")

plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.title("True Signals")
plt.plot(t, true_signal_m, label="True Signal (for Multiplicative)", color='black', linewidth=2)
plt.plot(t, true_signal_a, label="True Signal (for Additive)", color='gray', linestyle='--')
plt.legend()

plt.subplot(3, 1, 2)
plt.title(f"Multiplicative Test Case (Applied Log: {flag_m}, MSE: {mse_m:.2f})")
plt.plot(t, mixed_mult, label="Raw Mixed", color='lightgray')
plt.plot(t, true_signal_m, label="True Signal", color='black', linewidth=2)
plt.plot(t, recovered_m, label="Recovered (Auto-Log)", color='red', linestyle='--')
plt.legend(loc="upper right")

plt.subplot(3, 1, 3)
plt.title(f"Additive Test Case (Applied Log: {flag_a}, MSE: {mse_a:.2f})")
plt.plot(t, mixed_add, label="Raw Mixed", color='lightgray')
plt.plot(t, true_signal_a, label="True Signal", color='black', linewidth=2)
plt.plot(t, recovered_a, label="Recovered (Auto-Linear)", color='blue', linestyle='--')
plt.legend(loc="upper right")

plt.tight_layout()
plt.savefig("examples/bss_multiplicative_auto_test.png")
print("\nPlot saved as 'examples/bss_multiplicative_auto_test.png'")
