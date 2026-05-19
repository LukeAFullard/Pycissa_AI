import numpy as np
import pytest
from pycissa.processing.cissa.cissa import Cissa
from pycissa.processing.cissa.overlap_cissa import OverlapCissa

def test_overlap_cissa_basic_execution():
    N = 1000
    t = np.arange(N)
    x = np.sin(2 * np.pi * t / 100) + 0.1 * np.random.randn(N)

    Z = 200
    q = 100
    L = 50

    oc = OverlapCissa(t, x, Z=Z, q=q, L=L)
    oc.fit()

    # Assert final reconstructed components have exactly N rows
    assert oc.Z.shape[0] == N, f"Expected length {N}, got {oc.Z.shape[0]}"

    # Let's compare standard Cissa versus OverlapCissa shape
    cissa_std = Cissa(t, x)
    cissa_std.fit(L=L)

    assert oc.Z.shape[1] == cissa_std.Z.shape[1], "Number of components differs"

def test_overlap_cissa_edge_cases():
    N = 150 # Some length not perfectly divisible by q
    t = np.arange(N)
    x = np.sin(2 * np.pi * t / 50) + 0.1 * np.random.randn(N)

    # If q = 60, Z = 100, then L_bar = 20
    Z = 100
    q = 60
    L = 20

    oc = OverlapCissa(t, x, Z=Z, q=q, L=L)
    oc.fit()

    assert oc.Z.shape[0] == N

def test_overlap_cissa_post_processing():
    """
    Test that OverlapCissa properly inherits from Cissa and supports
    post-processing methods like post_run_monte_carlo_analysis and post_group_components.
    """
    np.random.seed(42)
    N = 200
    t = np.arange(N)
    x = np.sin(2 * np.pi * t / 20) + np.random.normal(0, 0.5, N)

    L = 20
    q = 40
    L_bar = 10
    Z_len = q + 2 * L_bar

    ocissa = OverlapCissa(t, x, Z=Z_len, q=q, L=L)
    ocissa.fit()

    # Run Monte Carlo significance testing
    ocissa.post_run_monte_carlo_analysis(alpha=0.05, K_surrogates=5, surrogates='random_permutation')

    # Run grouping based on monte_carlo
    ocissa.post_group_components(grouping_type='monte_carlo', plot_result=False)

    # Check that the components were correctly grouped
    assert hasattr(ocissa, 'x_trend')
    assert hasattr(ocissa, 'x_periodic')
    assert hasattr(ocissa, 'x_noise')
    assert len(ocissa.x_trend) == N
    assert 'monte_carlo_surrogate_type' in ocissa.results['cissa']['model parameters']
