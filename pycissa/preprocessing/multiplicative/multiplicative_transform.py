import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from typing import Optional, List, Tuple

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


class MultiplicativeTransformer:
    """
    A utility class to linearize multiplicative signals by applying a log-transform,
    managing any necessary positive offsets, and correctly inverting the transform later.
    This allows linear algorithms like CiSSA or M-CiSSA to process multiplicative mixtures.
    """

    def __init__(self):
        self.offsets = {}
        self.is_fitted = False

    def fit_transform(self, X: np.ndarray, columns_to_transform: Optional[List[int]] = None) -> np.ndarray:
        """
        Calculates necessary offsets to make data strictly positive and applies a natural log transform.

        Args:
            X (np.ndarray): The input data (1D or 2D array).
            columns_to_transform (list[int] | None): If X is 2D, a list of column indices to transform.
                                                     If None, transforms all columns (or the single 1D array).

        Returns:
            np.ndarray: The log-transformed array.
        """
        X_trans = X.copy()

        # Handle 1D
        if X_trans.ndim == 1:
            min_val = np.min(X_trans)
            offset = abs(min_val) + 1.0 if min_val <= 0 else 0.0
            self.offsets[0] = offset
            X_trans = np.log(X_trans + offset)
            self.is_fitted = True
            return X_trans

        # Handle 2D
        if columns_to_transform is None:
            columns_to_transform = list(range(X_trans.shape[1]))

        for col_idx in columns_to_transform:
            min_val = np.min(X_trans[:, col_idx])
            offset = abs(min_val) + 1.0 if min_val <= 0 else 0.0
            self.offsets[col_idx] = offset
            X_trans[:, col_idx] = np.log(X_trans[:, col_idx] + offset)

        self.is_fitted = True
        return X_trans

    def inverse_transform(self, X_transformed: np.ndarray, col_idx: int = 0) -> np.ndarray:
        """
        Inverts the log-transform back to the original linear scale.

        Args:
            X_transformed (np.ndarray): The 1D array of transformed components (e.g., recovered signal).
            col_idx (int, optional): The column index this 1D array corresponds to (defaults to 0 for 1D inputs).
                                     Used to retrieve the correct offset.

        Returns:
            np.ndarray: The exponentiated and offset-corrected array.
        """
        if not self.is_fitted:
            raise ValueError("The transformer must be fit before inverse_transform can be called.")

        if col_idx not in self.offsets:
            raise ValueError(f"Column index {col_idx} was not transformed during fit.")

        offset = self.offsets[col_idx]
        return np.exp(X_transformed) - offset
