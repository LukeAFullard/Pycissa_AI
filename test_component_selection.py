import numpy as np
import warnings
from pycissa.preprocessing.gap_fill.gap_filling import m_fill_timeseries_gaps

x = np.array([
    [1.0, 1.0], [1.1, 1.1], [1.2, 1.2], [1.3, 1.3], [1.4, 1.4], [1.5, 1.5], [1.6, 1.6], [1.7, 1.7],
    [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan],
    [1.8, 1.8], [1.9, 1.9], [2.0, 2.0], [2.1, 2.1], [2.2, 2.2], [2.3, 2.3]
])
t = np.arange(len(x))

try:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        results = m_fill_timeseries_gaps(t, x, L=3, test_number=1, estimate_error=True)
        for warn in w:
            if warn.category is not DeprecationWarning:
                print(warn.message)
except Exception as e:
    import traceback
    traceback.print_exc()
