import scipy.interpolate as si
original_interp1d = si.interp1d
import traceback
import sys

def spy_interp1d(x, y, *args, **kwargs):
    if len(x) != len(y):
        print(f"FAILED: x len = {len(x)}, y len = {len(y)}")
        traceback.print_stack(file=sys.stdout)
    return original_interp1d(x, y, *args, **kwargs)

si.interp1d = spy_interp1d
import numpy as np
import warnings
from pycissa.processing.mcissa.mcissa import MCissa

np.random.seed(42)
true_t = np.arange(15.2, 365.25 * 3, 365.25 / 12)
jitter = np.random.uniform(-14, 14, size=len(true_t))
observed_t = true_t + jitter

def generate_signals(t):
    trend1 = 0.01 * t
    annual1 = 5 * np.sin(2 * np.pi * t / 365.25)
    noise1 = np.random.normal(0, 0.5, len(t))
    return trend1 + annual1 + noise1

obs_x1 = generate_signals(observed_t)
obs_x = obs_x1.reshape(-1, 1)
obs_x[[5, 6, 18, 19, 20], :] = np.nan
dt = 365.25 / 12

try:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('ignore')
        mc_cissa = MCissa(observed_t, obs_x)
        mc_cissa.pre_fill_uneven_timeseries(dt=dt, L_values=[12], gap_threshold=20, test_number=0, multivariate=False, plot=False)

except Exception as e:
    pass
