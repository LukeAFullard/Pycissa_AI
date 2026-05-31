import numpy as np
from pycissa.utilities.bss_pipeline import auto_bss_pipeline

T = 400
t = np.arange(T)

true_signal_m = 10.0 + 3.0 * np.sin(2 * np.pi * t / 15.0)
artifact_m = 1.5 + 0.8 * np.sin(2 * np.pi * t / 60.0)
mixed_mult = true_signal_m * artifact_m + np.random.randn(T) * 0.1
ref_mult = artifact_m + np.random.randn(T) * 0.1

recovered, is_mult = auto_bss_pipeline(t, mixed_mult, ref_mult, K_surrogates=5, variance_threshold=0.01, alpha=1.0)
print(is_mult)
