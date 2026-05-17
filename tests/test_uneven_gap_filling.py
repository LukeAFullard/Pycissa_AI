import numpy as np
import pytest
from pycissa.preprocessing.gap_fill.uneven_gap_filling import fill_uneven_timeseries

def test_fill_uneven_timeseries_basic():
    # Generate some simple uneven data (e.g. a sine wave with some noise)
    np.random.seed(42)
    t = np.sort(np.random.uniform(0, 50, 40))
    x = np.sin(t) + np.random.normal(0, 0.05, 40)

    # Intentionally create a large gap
    gap_mask = (t > 20) & (t < 30)
    t = t[~gap_mask]
    x = x[~gap_mask]

    res = fill_uneven_timeseries(
        t=t,
        x=x,
        L_values=[5, 10],
        dt=1.0,
        gap_threshold=1.5,
        plot=False
    )

    assert 'best_L' in res
    assert res['best_L'] in [5, 10]
    assert 'rmse' in res
    assert 'r2' in res
    assert 't_even' in res
    assert 'x_even_filled' in res
    assert 'x_back_interp' in res
    assert 'Z_back_interp' in res
    assert res['Z_back_interp'] is not None
    # check shapes
    assert res['Z_back_interp'].shape[0] == len(res['x_back_interp'])

    # Ensure it works without errors and r2 is reasonable
    assert res["rmse"] is not None

def test_fill_uneven_timeseries_no_gaps_warning():
    t = np.arange(0, 100, 1.0)
    x = np.sin(t)

    with pytest.warns(UserWarning, match="No gaps found"):
        res = fill_uneven_timeseries(
            t=t,
            x=x,
            L_values=[5],
            dt=1.0,
            gap_threshold=2.0,
            plot=False
        )

def test_fill_uneven_timeseries_ccc():
    np.random.seed(42)
    t = np.sort(np.random.uniform(0, 50, 40))
    x = np.sin(t) + np.random.normal(0, 0.05, 40)

    gap_mask = (t > 20) & (t < 30)
    t = t[~gap_mask]
    x = x[~gap_mask]

    res = fill_uneven_timeseries(
        t=t,
        x=x,
        L_values=[5, 10],
        dt=1.0,
        gap_threshold=1.5,
        optimization_metric='ccc',
        plot=False
    )

    assert 'ccc' in res
    assert res['ccc'] > 0.0
    assert res['best_L'] in [5, 10]
