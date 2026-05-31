import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from typing import Tuple

def test_if_multiplicative(mixed: np.ndarray, ref: np.ndarray, window_size: int = 20, threshold_ratio: float = 0.5, p_value_thresh: float = 0.05) -> Tuple[bool, float, float]:
    """
    Auto-tests if an artifact or noise is likely multiplicative rather than additive.
    Multiplicative noise modulates the amplitude (envelope/variance) of the main signal.

    Args:
        mixed (np.ndarray): The 1D mixed signal array.
        ref (np.ndarray): The 1D reference signal array.
        window_size (int, optional): The rolling window size for variance calculation. Defaults to 20.
        threshold_ratio (float, optional): The threshold ratio of variance correlation vs raw correlation. Defaults to 0.5.
        p_value_thresh (float, optional): The p-value threshold for statistical significance. Defaults to 0.05.

    Returns:
        tuple[bool, float, float]: A tuple containing:
            - is_mult (bool): True if the artifact is likely multiplicative, False otherwise.
            - corr_raw (float): The Pearson correlation coefficient between the mixed signal and reference.
            - corr_std (float): The Pearson correlation coefficient between the rolling standard deviation of the mixed signal and the reference.
    """
    mixed_series = pd.Series(mixed)

    # Calculate local amplitude (rolling standard deviation)
    rolling_std = mixed_series.rolling(window=window_size, center=True).std().bfill().ffill().values

    # Calculate correlations
    corr_raw, p_raw = pearsonr(mixed, ref)
    corr_std, p_std = pearsonr(rolling_std, ref)

    is_mult = False
    if p_std < p_value_thresh and abs(corr_std) > 0.2:
        if abs(corr_raw) < 0.01:
            is_mult = True
        elif (abs(corr_std) / abs(corr_raw)) > threshold_ratio:
            is_mult = True

    return is_mult, corr_raw, corr_std
