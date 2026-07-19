import numpy as np
from pycissa import Cissa
import warnings

def test_pre_fill_uneven_timeseries_update_state_true():
    # Large offset
    t = np.array([1.0, 2.5, 3.8, 5.9, 7.2, 8.5, 9.8, 11.1, 12.5, 13.8])
    x = np.array([10.0, 25.0, 38.0, 59.0, 72.0, 85.0, 98.0, 111.0, 125.0, 138.0])

    cissa = Cissa(t, x)
    # Gap filling works best when outliers=['nan_only', None] is used
    cissa.pre_fill_uneven_timeseries(
        L_values=[3],
        dt=1.0,
        gap_threshold=0.5,
        update_state=True,
        plot=False,
        outliers=['nan_only', None]
    )

    assert len(cissa.t) > 0
    # Expected grid based on min/max of t and dt=1.0
    expected_t = np.arange(1.0, 13.8 + 1.0, 1.0)
    np.testing.assert_array_almost_equal(cissa.t, expected_t)
    assert cissa.x.shape == cissa.t.shape

def test_pre_fill_uneven_timeseries_update_state_false():
    # Large offset
    t = np.array([1.0, 2.5, 3.8, 5.9, 7.2, 8.5, 9.8, 11.1, 12.5, 13.8])
    x = np.array([10.0, 25.0, 38.0, 59.0, 72.0, 85.0, 98.0, 111.0, 125.0, 138.0])

    cissa = Cissa(t, x)
    original_t = cissa.t.copy()
    original_x = cissa.x.copy()

    cissa.pre_fill_uneven_timeseries(
        L_values=[3],
        dt=1.0,
        gap_threshold=0.5,
        update_state=False,
        plot=False,
        outliers=['nan_only', None]
    )

    # Check that t and x remain exactly the original arrays
    np.testing.assert_array_almost_equal(cissa.t, original_t)
    np.testing.assert_array_almost_equal(cissa.x, original_x)

    # But uneven_gap_fill_results should be present in cissa
    assert hasattr(cissa, 'uneven_gap_fill_results')
    assert 't_even' in cissa.uneven_gap_fill_results

def test_pre_fill_uneven_timeseries_with_gaps():
    # Offset with gaps (e.g. missing middle entirely)
    t = np.array([1.1, 2.2, 3.1, 7.2, 8.1, 9.2, 10.1, 11.2, 12.1])
    x = np.array([11.0, 22.0, 31.0, 72.0, 81.0, 92.0, 101.0, 112.0, 121.0])

    cissa = Cissa(t, x)
    cissa.pre_fill_uneven_timeseries(
        L_values=[3],
        dt=1.0,
        gap_threshold=0.5,
        update_state=True,
        plot=False,
        outliers=['nan_only', None]
    )

    expected_t = np.arange(1.1, 12.1 + 1.0, 1.0)
    np.testing.assert_array_almost_equal(cissa.t, expected_t)

    # x should have no NaNs since they were filled
    assert not np.any(np.isnan(cissa.x))
    assert len(cissa.x) == len(expected_t)
