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
