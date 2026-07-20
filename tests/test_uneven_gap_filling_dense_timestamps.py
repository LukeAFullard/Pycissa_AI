import numpy as np
import pytest
import warnings
from pycissa.preprocessing.gap_fill.uneven_gap_filling import fill_uneven_timeseries, m_fill_uneven_timeseries

def test_fill_uneven_timeseries_dense_timestamps_with_missing_block():
    t = np.arange(0, 300, 1.0)
    x = np.sin(2 * np.pi * t / 24)
    x[100:130] = np.nan

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = fill_uneven_timeseries(t, x, L_values=[24], dt=1.0, gap_threshold=2.0, plot=False)

        has_nans_in_gaps = np.any(np.isnan(res['x_even_with_gaps']))
        assert has_nans_in_gaps, "Gap was not detected in dense timestamp array!"

        for warn in w:
            if "No gaps found based on the given gap_threshold" in str(warn.message):
                pytest.fail("'No gaps found' warning incorrectly triggered despite present missing block.")

def test_m_fill_uneven_timeseries_dense_timestamps_with_missing_block():
    t = np.arange(0, 300, 1.0)
    x = np.sin(2 * np.pi * t / 24)
    x_multi = np.column_stack((x, x))
    x_multi[100:130, :] = np.nan

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = m_fill_uneven_timeseries(t, x_multi, L_values=[24], dt=1.0, gap_threshold=2.0, plot=False)

        has_nans_in_gaps = np.any(np.isnan(res['x_even_with_gaps']))
        assert has_nans_in_gaps, "Gap was not detected in dense multivariate timestamp array!"

        for warn in w:
            if "No gaps found based on the given gap_threshold" in str(warn.message):
                pytest.fail("'No gaps found' warning incorrectly triggered despite present missing block in multi-channel array.")
