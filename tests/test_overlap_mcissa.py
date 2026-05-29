import numpy as np
import pytest
from pycissa.processing.mcissa.mcissa import MCissa
from pycissa.processing.mcissa.overlap_mcissa import OverlapMCissa

def test_overlap_mcissa_basic_execution():
    N = 1000
    M = 2
    t = np.arange(N)

    # Generate 2-channel time series
    x1 = np.sin(2 * np.pi * t / 100) + 0.1 * np.random.randn(N)
    x2 = np.cos(2 * np.pi * t / 100) + 0.1 * np.random.randn(N)
    x = np.column_stack((x1, x2))

    Z = 200
    q = 100
    L = 50

    omc = OverlapMCissa(t, x, Z=Z, q=q, L=L)
    omc.fit()

    # Calculate expected nft length
    # Note: MCissa generates grouping frequencies from np.zeros(L)
    nft = omc.Z_stacked.shape[2]

    # Assert final reconstructed components have correct shape (T_original, M, nft)
    assert omc.Z_stacked.shape == (N, M, nft), f"Expected shape {(N, M, nft)}, got {omc.Z_stacked.shape}"

    # Assert Zs is a list of length M and each array has the correct shape
    assert len(omc.Zs) == M, f"Expected length {M}, got {len(omc.Zs)}"
    assert omc.Zs[0].shape == (N, M, nft), f"Expected shape {(N, M, nft)}, got {omc.Zs[0].shape}"
    assert omc.Zs[1].shape == (N, M, nft), f"Expected shape {(N, M, nft)}, got {omc.Zs[1].shape}"

    # Verify standard MCissa dictionary
    assert 'mcissa' in omc.results
    assert 'components' in omc.results['mcissa']
    assert omc.results['mcissa']['model parameters']['L'] == L

def test_overlap_mcissa_edge_cases():
    N = 150 # Some length not perfectly divisible by q
    M = 3
    t = np.arange(N)
    x1 = np.sin(2 * np.pi * t / 50) + 0.1 * np.random.randn(N)
    x2 = np.cos(2 * np.pi * t / 50) + 0.1 * np.random.randn(N)
    x3 = np.sin(2 * np.pi * t / 25) + 0.1 * np.random.randn(N)
    x = np.column_stack((x1, x2, x3))

    # If q = 60, Z = 100, then L_bar = 20
    Z = 100
    q = 60
    L = 20

    omc = OverlapMCissa(t, x, Z=Z, q=q, L=L)
    omc.fit()

    nft = omc.Z_stacked.shape[2]

    assert omc.Z_stacked.shape == (N, M, nft)
    assert len(omc.Zs) == M
    assert omc.Zs[0].shape == (N, M, nft)

def test_overlap_mcissa_post_processing():
    """
    Test that OverlapMCissa properly inherits from MCissa and supports
    post-processing methods like post_run_monte_carlo_analysis and post_group_components.
    """
    np.random.seed(42)
    N = 200
    M = 2
    t = np.arange(N)
    x1 = np.sin(2 * np.pi * t / 20) + np.random.normal(0, 0.5, N)
    x2 = np.cos(2 * np.pi * t / 20) + np.random.normal(0, 0.5, N)
    x = np.column_stack((x1, x2))

    L = 20
    q = 40
    L_bar = 10
    Z_len = q + 2 * L_bar

    omc = OverlapMCissa(t, x, Z=Z_len, q=q, L=L)
    omc.fit()

    # Run Monte Carlo significance testing
    omc.post_run_monte_carlo_analysis(alpha=0.05, K_surrogates=5, surrogates='random_permutation')

    # Run grouping based on monte_carlo
    omc.post_group_components(grouping_type='monte_carlo', plot_result=False)

    # Check that the components were correctly grouped
    assert hasattr(omc, 'x_trend')
    assert hasattr(omc, 'x_periodic')
    assert hasattr(omc, 'x_noise')
    assert omc.x_trend.shape == (N, M)
    assert omc.x_periodic.shape == (N, M)
    assert omc.x_noise.shape == (N, M)
    assert 'monte_carlo_surrogate_type' in omc.results['mcissa']['model parameters']
