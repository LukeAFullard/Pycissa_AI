import numpy as np
import pytest
from pycissa.processing.mcissa.mcissa import MCissa

def generate_synthetic_data(T=200):
    t = np.arange(1, T + 1)

    # Variable 1
    trend1 = 0.05 * t
    periodic1 = 2 * np.sin(2 * np.pi * t / 12) + 1.5 * np.cos(2 * np.pi * t / 6)
    noise1 = np.random.normal(0, 0.1, T)
    x1 = trend1 + periodic1 + noise1

    # Variable 2
    trend2 = 0.03 * t
    periodic2 = 1.5 * np.sin(2 * np.pi * t / 12 + np.pi/4) + 1.0 * np.cos(2 * np.pi * t / 6 - np.pi/3)
    noise2 = np.random.normal(0, 0.1, T)
    x2 = trend2 + periodic2 + noise2

    X = np.column_stack((x1, x2))
    return t, X

def test_mcissa_shapes_and_reconstruction():
    np.random.seed(42)
    T = 200
    L = 24
    M = 2
    t, X = generate_synthetic_data(T)

    mcissa = MCissa(t, X)

    # Test 'NoExt' to verify perfect reconstruction over the original domain
    mcissa.fit(L=L, extension_type='NoExt')

    # Check shapes
    Z_stacked = mcissa.Z_stacked
    # Calculate expected number of frequencies
    nf2 = (L + 1) // 2 - 1 if L % 2 != 0 else L // 2 - 1
    nft = nf2 + abs(L % 2 - 2)

    assert Z_stacked.shape == (T, M, nft), f"Expected shape {(T, M, nft)}, got {Z_stacked.shape}"
    assert mcissa.psd.shape == (L, M), f"Expected psd shape {(L, M)}, got {mcissa.psd.shape}"

    # Perfect reconstruction check
    # Summing over all frequencies for each variable should equal the original data
    X_reconstructed = np.sum(Z_stacked, axis=2)

    # The reconstruction should be very close to original
    np.testing.assert_allclose(X, X_reconstructed, rtol=1e-5, atol=1e-5)

def test_mcissa_eigenvalues():
    np.random.seed(42)
    T = 100
    L = 12
    t, X = generate_synthetic_data(T)

    mcissa = MCissa(t, X)
    mcissa.fit(L=L, extension_type='AR_LR')

    # Eigenvalues should be real and positive
    assert np.all(np.isreal(mcissa.psd)), "Eigenvalues should be real"
    assert np.all(mcissa.psd >= -1e-10), "Eigenvalues should be non-negative"

def test_mcissa_extension_ar():
    np.random.seed(42)
    T = 100
    L = 10
    M = 2
    t, X = generate_synthetic_data(T)

    mcissa = MCissa(t, X)
    # Using AR extension should produce valid output
    mcissa.fit(L=L, extension_type='AR_LR')

    # Due to AR extension, Z_stacked has no exact boundary matching unless we account for actual_left_ext and right_ext clipping.
    # The function truncates the output back to original size if extend_left and extend_right are correct.
    assert mcissa.Z_stacked.shape[0] == T


def test_mcissa_signal_extraction():
    np.random.seed(42)
    T = 200
    L = 24
    M = 2

    t = np.arange(1, T + 1)

    # Simple test: only periodic components + noise. No trend.
    periodic1 = 2 * np.sin(2 * np.pi * t / 12)
    noise1 = np.random.normal(0, 0.1, T)
    x1 = periodic1 + noise1

    periodic2 = 1.5 * np.sin(2 * np.pi * t / 12 + np.pi/4)
    noise2 = np.random.normal(0, 0.1, T)
    x2 = periodic2 + noise2

    X = np.column_stack((x1, x2))

    mcissa = MCissa(t, X)
    mcissa.fit(L=L, extension_type='NoExt')

    # Find the frequency index with the highest power
    # We can sum the eigenvalues (psd) across the M variables for each frequency
    # psd shape is (L, M)
    nf2 = (L + 1) // 2 - 1 if L % 2 != 0 else L // 2 - 1

    # We expect the largest eigenvalue to correspond to the period 12
    # The actual grouping depends on how the frequencies are sorted and grouped, but typically
    # the dominant signal will be isolated in one of the first few reconstructed components.

    Z_stacked = mcissa.Z_stacked

    # Sum the variance of the reconstructed components
    variances = [np.var(Z_stacked[:, 0, i]) + np.var(Z_stacked[:, 1, i]) for i in range(Z_stacked.shape[2])]
    dominant_idx = np.argmax(variances)

    dominant_signal = Z_stacked[:, :, dominant_idx]

    # The dominant signal should match the original periodic signals very closely
    np.testing.assert_allclose(dominant_signal[:, 0], periodic1, rtol=0.1, atol=0.2)
    np.testing.assert_allclose(dominant_signal[:, 1], periodic2, rtol=0.1, atol=0.2)

def test_mcissa_complex_signal_extraction():
    """
    Test extraction of 4 subsignals: a linear trend, a low-frequency wave,
    a high-frequency wave, and random noise.
    """
    np.random.seed(42)
    T = 300
    L = 36  # Needs to be large enough to capture the lowest frequency
    M = 3   # 3 variables

    t = np.arange(1, T + 1)

    # Subsignal 1: Trend
    trend1 = 0.05 * t
    trend2 = -0.02 * t
    trend3 = 0.01 * t

    # Subsignal 2: Low-frequency wave (Period = 24)
    low_freq1 = 3.0 * np.sin(2 * np.pi * t / 24)
    low_freq2 = 2.5 * np.cos(2 * np.pi * t / 24)
    low_freq3 = 1.0 * np.sin(2 * np.pi * t / 24 + np.pi / 4)

    # Subsignal 3: High-frequency wave (Period = 6)
    high_freq1 = 1.5 * np.sin(2 * np.pi * t / 6)
    high_freq2 = 1.5 * np.sin(2 * np.pi * t / 6 - np.pi / 3)
    high_freq3 = 2.0 * np.cos(2 * np.pi * t / 6)

    # Subsignal 4: Noise
    noise1 = np.random.normal(0, 0.1, T)
    noise2 = np.random.normal(0, 0.1, T)
    noise3 = np.random.normal(0, 0.1, T)

    x1 = trend1 + low_freq1 + high_freq1 + noise1
    x2 = trend2 + low_freq2 + high_freq2 + noise2
    x3 = trend3 + low_freq3 + high_freq3 + noise3

    X = np.column_stack((x1, x2, x3))

    mcissa = MCissa(t, X)
    mcissa.fit(L=L, extension_type='NoExt')

    Z_stacked = mcissa.Z_stacked

    # We expect 4 main components to be identifiable if we group by variance
    # Z_stacked has shape (T, M, nft)
    # We can measure the total variance across variables for each frequency
    variances = [np.sum([np.var(Z_stacked[:, m, i]) for m in range(M)]) for i in range(Z_stacked.shape[2])]

    # Sort frequencies by descending variance
    sorted_indices = np.argsort(variances)[::-1]

    # Extract the top 3 structural components (Trend, Low-freq, High-freq)
    # The trend usually dominates, followed by the highest amplitude periodic signal.
    comp_a = Z_stacked[:, :, sorted_indices[0]]
    comp_b = Z_stacked[:, :, sorted_indices[1]]
    comp_c = Z_stacked[:, :, sorted_indices[2]]

    # Let's check which component matches which subsignal
    def get_max_correlation(target, sorted_indices, Z_stacked):
        corrs = []
        # Usually the subsignals will be distributed among the top several components
        # (especially if the low-frequency wave is split into multiple subcomponents).
        # We find the sum of the best 2 contiguous components for the low-frequency, or just checking max corr
        for idx in sorted_indices[:5]:
            comp = Z_stacked[:, :, idx]
            corr = np.corrcoef(comp.flatten(), target.flatten())[0, 1]
            corrs.append(corr)

        # Also check combined top 2 components for low_freq which might be split
        combined_comp = Z_stacked[:, :, sorted_indices[2]] + Z_stacked[:, :, sorted_indices[3]]
        combined_corr = np.corrcoef(combined_comp.flatten(), target.flatten())[0, 1]
        corrs.append(combined_corr)

        return max(corrs)

    target_trend = np.column_stack((trend1, trend2, trend3))
    target_low = np.column_stack((low_freq1, low_freq2, low_freq3))
    target_high = np.column_stack((high_freq1, high_freq2, high_freq3))

    matched_trend = get_max_correlation(target_trend, sorted_indices, Z_stacked) > 0.95
    matched_low = get_max_correlation(target_low, sorted_indices, Z_stacked) > 0.95
    matched_high = get_max_correlation(target_high, sorted_indices, Z_stacked) > 0.95

    assert matched_trend, "Failed to accurately extract the trend subsignal."
    assert matched_low, "Failed to accurately extract the low-frequency subsignal."
    assert matched_high, "Failed to accurately extract the high-frequency subsignal."
