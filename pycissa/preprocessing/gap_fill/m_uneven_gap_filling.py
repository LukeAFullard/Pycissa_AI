import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.interpolate import interp1d
from pycissa.preprocessing.gap_fill.gap_filling import m_fill_timeseries_gaps

def m_fill_uneven_timeseries(t: np.ndarray,
                             x: np.ndarray,
                             L_values: list[int],
                             dt: float,
                             gap_threshold: float,
                             eps_values: list[float] = None,
                             interp_method: str = 'cubic',
                             optimization_metric: str = 'rmse',
                             r2_warning_threshold: float = 0.5,
                             plot: bool = True,
                             **kwargs) -> dict:
    """
    Fills gaps in a multivariate unevenly sampled time series by interpolating to an even grid,
    applying joint gap filling while optimizing the window length L, and projecting back.
    """

    # 1. Create even grid
    t_even = np.arange(np.nanmin(t), np.nanmax(t) + dt, dt)

    # Interpolate x onto t_even for initial guess
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    valid_mask = ~np.isnan(x).any(axis=1)
    interpolator = interp1d(t[valid_mask], x[valid_mask], kind=interp_method, bounds_error=False, fill_value="extrapolate", axis=0)
    x_even_interp = interpolator(t_even)

    # 2. Identify gaps
    t_valid = t[valid_mask]

    if len(t_valid) > 0:
        idx = np.searchsorted(t_valid, t_even)
        idx = np.clip(idx, 1, len(t_valid) - 1)
        left_dist = np.abs(t_even - t_valid[idx - 1])
        right_dist = np.abs(t_even - t_valid[idx])
        min_distances = np.minimum(left_dist, right_dist)
    else:
        min_distances = np.full_like(t_even, np.inf)

    x_even_with_gaps = x_even_interp.copy()
    gaps_mask = min_distances > gap_threshold
    x_even_with_gaps[gaps_mask, :] = np.nan

    if not np.any(gaps_mask):
        warnings.warn("No gaps found based on the given gap_threshold. The entire even grid is considered known data.")

    if eps_values is None:
        eps_values = [1.0]

    best_L = None
    best_eps = None
    best_x_even_filled = None
    best_metric_val = np.inf if optimization_metric.lower() == 'rmse' else -np.inf
    best_rmse = np.inf
    best_r2 = -np.inf
    best_ccc = -np.inf
    best_x_back_interp = None

    results = {}

    if optimization_metric.lower() not in ['rmse', 'ccc']:
        raise ValueError("optimization_metric must be either 'rmse' or 'ccc'")

    # 3. Optimization Loop
    for L in L_values:
      for eps in eps_values:
        try:
            # m_fill_timeseries_gaps returns: x_ca, error_estimates, error_estimates_percentage, error_rmse, error_rmse_percentage, original_points, imputed_points, fig_errors, fig_time_series
            res = m_fill_timeseries_gaps(t=t_even, x=x_even_with_gaps, L=L, convergence=['value', eps], **kwargs)
            if isinstance(res, tuple):
                x_even_filled = res[0]
            else:
                x_even_filled = res

            from pycissa.processing.matrix_operations.m_matrix_operations import run_mcissa
            try:
                Z_stacked, _, _ = run_mcissa(x_even_filled, L, extension_type=kwargs.get('extension_type', 'AR_LR'), multi_thread_run=kwargs.get('multi_thread_run', True))
            except Exception:
                Z_stacked = x_even_filled[:, :, np.newaxis]

            # Z_stacked has shape (T, M, nft)
            # Sum across components (nft) for each channel to get back to (T, M)
            Z_back_interp = np.zeros((len(t), x.shape[1], Z_stacked.shape[2]))

            for m in range(x.shape[1]):
                for i in range(Z_stacked.shape[2]):
                    comp_interpolator = interp1d(t_even, Z_stacked[:, m, i], kind=interp_method, bounds_error=False, fill_value="extrapolate")
                    Z_back_interp[:, m, i] = comp_interpolator(t)

            x_back_interp = np.sum(Z_back_interp, axis=2)

            # Calculate metrics
            valid_mask_eval = ~np.isnan(x).any(axis=1) & ~np.isnan(x_back_interp).any(axis=1)
            if np.sum(valid_mask_eval) < 2:
                continue

            y_true = x[valid_mask_eval].flatten()
            y_pred = x_back_interp[valid_mask_eval].flatten()
            rmse = np.sqrt(np.mean((y_true - y_pred)**2))
            ss_res = np.sum((y_true - y_pred)**2)
            ss_tot = np.sum((y_true - np.mean(y_true))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

            # Calculate Concordance Correlation Coefficient (CCC)
            mean_true = np.mean(y_true)
            mean_pred = np.mean(y_pred)
            var_true = np.var(y_true)
            var_pred = np.var(y_pred)
            covar = np.cov(y_true, y_pred, ddof=0)[0, 1] if len(y_true) > 1 else 0.0
            ccc = (2 * covar) / (var_true + var_pred + (mean_true - mean_pred)**2) if (var_true + var_pred) > 0 else 0.0

            # Determine if this L is better
            is_better = False
            if optimization_metric.lower() == 'rmse':
                is_better = rmse < best_metric_val
                if is_better: best_metric_val = rmse
            elif optimization_metric.lower() == 'ccc':
                is_better = ccc > best_metric_val
                if is_better: best_metric_val = ccc

            if is_better:
                best_rmse = rmse
                best_r2 = r2
                best_ccc = ccc
                best_L = L
                best_eps = eps
                best_x_even_filled = x_even_filled
                best_x_back_interp = x_back_interp

        except Exception as e:
            warnings.warn(f"Gap filling failed for L={L}, eps={eps} with error: {str(e)}")
            continue

    if best_L is None:
        raise ValueError("Could not find a successful L from L_values.")

    # 4. Warnings
    if best_r2 < r2_warning_threshold:
        warnings.warn(f"Poor fit detected for best L={best_L}. R-squared = {best_r2:.4f} < {r2_warning_threshold}")

    ret_dict = {
        'best_L': best_L,
        'best_eps': best_eps,
        'rmse': best_rmse,
        'r2': best_r2,
        'ccc': best_ccc,
        't_even': t_even,
        'x_even_with_gaps': x_even_with_gaps,
        'x_even_filled': best_x_even_filled,
        't_uneven': t,
        'x_uneven': x,
        'x_back_interp': best_x_back_interp,
        'gaps_mask': gaps_mask
    }
    return ret_dict
