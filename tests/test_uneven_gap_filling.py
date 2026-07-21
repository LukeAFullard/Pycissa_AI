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

def test_fill_uneven_timeseries_poor_fit_fallback():
    # Generate data designed to fail poorly with cubic interpolation across a gap
    np.random.seed(42)
    t = np.sort(np.random.uniform(0, 50, 40))
    x = np.sin(t) + np.random.normal(0, 0.05, 40)
    gap_mask = (t > 20) & (t < 30)
    t = t[~gap_mask]
    x = x[~gap_mask]

    # This should internally trigger the fallback to linear and avoid warning/bad R2
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = fill_uneven_timeseries(
            t=t,
            x=x,
            L_values=[5, 10],
            dt=1.0,
            gap_threshold=1.5,
            interp_method='cubic',
            plot=False
        )
        for warn in w:
            if "Poor fit" in str(warn.message):
                pytest.fail(f"Poor fit warning was not suppressed by fallback: {warn.message}")

    assert res['r2'] > 0.5


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

def test_fill_uneven_timeseries_monte_carlo():
    np.random.seed(42)
    t = np.sort(np.random.uniform(0, 50, 40))
    x = np.sin(t) + np.random.normal(0, 0.05, 40)

    gap_mask = (t > 20) & (t < 30)
    t = t[~gap_mask]
    x = x[~gap_mask]

    res = fill_uneven_timeseries(
        t=t,
        x=x,
        L_values=[5],
        dt=1.0,
        gap_threshold=1.5,
        component_selection_method='monte_carlo_significant_components',
        alpha=0.05,
        surrogates='random_permutation',
        K_surrogates=10, # low number for speed
        plot=False
    )

    assert res['best_L'] == 5
    assert res['r2'] > 0.0

from pycissa.preprocessing.gap_fill.uneven_gap_filling import m_fill_uneven_timeseries

def test_m_fill_uneven_timeseries_monte_carlo():
    np.random.seed(42)
    t = np.sort(np.random.uniform(0, 50, 40))
    x = np.column_stack([np.sin(t) + np.random.normal(0, 0.05, 40), np.cos(t) + np.random.normal(0, 0.05, 40)])

    gap_mask = (t > 20) & (t < 30)
    t = t[~gap_mask]
    x = x[~gap_mask]

    res = m_fill_uneven_timeseries(
        t=t,
        x=x,
        L_values=[5],
        dt=1.0,
        gap_threshold=1.5,
        component_selection_method='monte_carlo_significant_components',
        alpha=0.05,
        surrogates='random_permutation',
        K_surrogates=10, # low number for speed
        plot=False
    )

    assert res['best_L'] == 5
    assert res['r2'] > 0.0
def test_fill_uneven_timeseries_Z_back_interp_consistency():
    """
    Test that Z_back_interp matches the back-interpolated x for the best_L found,
    preventing the bug where Z_back_interp tracked the last evaluated L in the loop
    instead of best_L. (Issue 6)
    """
    from pycissa.preprocessing.gap_fill.uneven_gap_filling import fill_uneven_timeseries
    import numpy as np

    t = np.arange(0, 300, 1.0)
    x = np.sin(2*np.pi*t/24) + 0.03*np.random.randn(300)
    x[100:110] = np.nan

    res = fill_uneven_timeseries(t, x, L_values=[24, 13], dt=1.0, gap_threshold=2.0, plot=False)

    assert res['best_L'] == 24, "Expected L=24 to be the best fit."
    mismatch = np.nanmax(np.abs(np.sum(res['Z_back_interp'], axis=1) - res['x_back_interp']))
    assert mismatch < 1e-10, f"Z_back_interp is inconsistent with x_back_interp. Mismatch: {mismatch}"

def test_m_fill_uneven_timeseries_misaligned_gaps():
    """
    Test that multivariate gap filling successfully detects gaps when they are
    misaligned across channels. Previously, a gap would only trigger if ALL channels
    were missing data at a specific timestamp. (Issue 7)
    """
    from pycissa.preprocessing.gap_fill.uneven_gap_filling import m_fill_uneven_timeseries
    import numpy as np

    t = np.arange(0, 300, 1.0)
    x1 = np.sin(2*np.pi*t/24) + 0.05*np.random.randn(300)
    x2 = np.cos(2*np.pi*t/24) + 0.05*np.random.randn(300)
    x = np.column_stack([x1, x2])

    # Only channel 0 is missing; channel 1 stays valid
    x[100:130, 0] = np.nan

    # We do NOT expect a warning about "No gaps found" because channel 0 has a gap.
    # Therefore, we expect x_even_with_gaps to contain NaNs for channel 0 at those indices.
    res = m_fill_uneven_timeseries(t, x, L_values=[24], dt=1.0, gap_threshold=2.0, plot=False)

    has_gaps_in_interp = np.any(np.isnan(res['x_even_with_gaps']))
    assert has_gaps_in_interp, "Gap was not correctly identified and masked with NaNs before spectral filling."
