import numpy as np
from pycissa.preprocessing import test_if_multiplicative as run_multiplicative_test

def test_variance_correlation_test():
    # 1. Multiplicative
    np.random.seed(42)
    T = 400
    t = np.arange(T)
    true_signal = 10.0 + 3.0 * np.sin(2 * np.pi * t / 15.0)
    artifact = 1.5 + 0.8 * np.sin(2 * np.pi * t / 60.0)
    mixed_mult = true_signal * artifact + np.random.randn(T) * 0.1
    ref = artifact + np.random.randn(T) * 0.1

    is_mult, _, _ = run_multiplicative_test(mixed_mult, ref)
    assert is_mult == True

    # 2. Additive
    mixed_add = true_signal + artifact * 5.0 + np.random.randn(T) * 0.1
    is_mult_add, _, _ = run_multiplicative_test(mixed_add, ref)
    assert is_mult_add == False
