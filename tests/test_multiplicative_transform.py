import numpy as np
import pytest
from pycissa.preprocessing import test_if_multiplicative as run_multiplicative_test, MultiplicativeTransformer

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

def test_multiplicative_transformer_1d():
    np.random.seed(42)
    T = 100
    t = np.arange(T)
    # Signal with negative values
    X_1d = np.sin(t) * 5.0 - 2.0

    transformer = MultiplicativeTransformer()
    X_log = transformer.fit_transform(X_1d)

    assert transformer.is_fitted == True
    assert 0 in transformer.offsets
    # Min is approx -7. Offset should be > 7.
    assert transformer.offsets[0] > 7.0

    X_rec = transformer.inverse_transform(X_log, col_idx=0)
    assert np.allclose(X_1d, X_rec)

def test_multiplicative_transformer_2d():
    np.random.seed(42)
    T = 100
    t = np.arange(T)
    # Col 0 has negatives, Col 1 is all positive
    X_2d = np.column_stack([np.sin(t)*5 - 2, np.cos(t)*10 + 20])

    trans2 = MultiplicativeTransformer()
    # Transform only column 0
    X_log_partial = trans2.fit_transform(X_2d, columns_to_transform=[0])
    assert 0 in trans2.offsets
    assert 1 not in trans2.offsets

    X_rec_0 = trans2.inverse_transform(X_log_partial[:, 0], col_idx=0)
    assert np.allclose(X_2d[:, 0], X_rec_0)

    # Transform all columns
    trans_all = MultiplicativeTransformer()
    X_log_all = trans_all.fit_transform(X_2d)
    assert 0 in trans_all.offsets
    assert 1 in trans_all.offsets
    # Col 1 minimum is approx 10. Offset should be 0 because it's already > 0.
    assert trans_all.offsets[1] == 0.0

    X_rec_all_1 = trans_all.inverse_transform(X_log_all[:, 1], col_idx=1)
    assert np.allclose(X_2d[:, 1], X_rec_all_1)

def test_multiplicative_transformer_unfitted_error():
    transformer = MultiplicativeTransformer()
    with pytest.raises(ValueError):
        transformer.inverse_transform(np.array([1, 2, 3]))

def test_multiplicative_transformer_invalid_col_error():
    np.random.seed(42)
    X = np.random.randn(10)
    transformer = MultiplicativeTransformer()
    X_log = transformer.fit_transform(X)
    with pytest.raises(ValueError):
        transformer.inverse_transform(X_log, col_idx=1) # 1 doesn't exist
