import numpy as np
import pytest
from pycissa.preprocessing.gap_fill.uneven_gap_filling import fill_uneven_timeseries, m_fill_uneven_timeseries
import warnings

def test_fill_uneven_timeseries_extreme_poor_fit():
    np.random.seed(42)
    # create extreme sparsity that even linear might be poor for
    t = np.sort(np.random.uniform(0, 50, 40))
    x = np.sin(t) + np.random.normal(0, 5.0, 40)
    gap_mask = (t > 20) & (t < 30)
    t = t[~gap_mask]
    x = x[~gap_mask]

    with pytest.warns(UserWarning, match="Poor fit detected for best L"):
        res = fill_uneven_timeseries(
            t=t,
            x=x,
            L_values=[2],
            dt=1.0,
            gap_threshold=5.0,
            interp_method='pchip',
            plot=False,
            r2_warning_threshold=0.9
        )
        assert res['r2'] < 0.9

def test_m_fill_uneven_timeseries_extreme_poor_fit():
    np.random.seed(42)
    # create extreme sparsity that even linear might be poor for
    t = np.sort(np.random.uniform(0, 50, 40))
    x = np.sin(t) + np.random.normal(0, 5.0, 40)
    gap_mask = (t > 20) & (t < 30)
    t = t[~gap_mask]
    x = x[~gap_mask]
    # make it multivariate
    x = np.column_stack([x, x * 2])

    with pytest.warns(UserWarning, match="Poor fit detected for best L"):
        res = m_fill_uneven_timeseries(
            t=t,
            x=x,
            L_values=[2],
            dt=1.0,
            gap_threshold=5.0,
            interp_method='pchip',
            plot=False,
            r2_warning_threshold=0.9
        )
        assert res['r2'] < 0.9
