import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.interpolate import interp1d
from pycissa.preprocessing.gap_fill.gap_filling import fill_timeseries_gaps

def fill_uneven_timeseries(t: np.ndarray,
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
    Fills gaps in an unevenly sampled time series by interpolating to an even grid,
    applying gap filling while optimizing the window length L, and projecting back.

    Parameters
    ----------
    t : np.ndarray
        Unevenly sampled time points.
    x : np.ndarray
        Values at uneven time points.
    L_values : list[int]
        List of L (window length) parameters to optimize over for CiSSA gap filling.
    dt : float
        Grid spacing for the evenly sampled grid.
    gap_threshold : float
        Max distance to a real data point on the even grid before it is considered a gap (NaN).
    eps_values : list[float], optional
        List of convergence epsilon values to optimize over. If None, defaults to [1.0].
    interp_method : str, optional
        Interpolation method ('linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', or 'next'). The default is 'cubic'.
    optimization_metric : str, optional
        Metric to optimize when selecting L. Either 'rmse' (minimize Root Mean Squared Error) or 'ccc' (maximize Concordance Correlation Coefficient). The default is 'rmse'.
    r2_warning_threshold : float, optional
        R-squared threshold below which a warning is issued. Default is 0.5.
    plot : bool, optional
        Whether to produce diagnostic plots.
    **kwargs :
        Additional arguments passed to fill_timeseries_gaps.

    Returns
    -------
    dict
        Dictionary containing the best L, original and filled data, and statistics.
    """

    # 1. Create even grid
    t_even = np.arange(np.nanmin(t), np.nanmax(t) + dt, dt)

    # Interpolate x onto t_even for initial guess
    # We use interp1d, handling values outside the range by extrapolating or bounding
    if x.ndim > 1:
        valid_mask = ~np.isnan(x).any(axis=1)
    else:
        valid_mask = ~np.isnan(x)

    interpolator = interp1d(t[valid_mask], x[valid_mask], kind=interp_method, bounds_error=False, fill_value="extrapolate", axis=0)
    x_even_interp = interpolator(t_even)

    # 2. Identify gaps
    # For each point in t_even, find the distance to the closest point in t that has valid (non-NaN) data
    # If distance > gap_threshold, set x_even_interp to NaN
    # We use searchsorted for efficient nearest neighbor distance calculation
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
    x_even_with_gaps[gaps_mask] = np.nan

    # If no gaps are found based on the threshold, warn the user
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
            # fill_timeseries_gaps returns: x_ca, out, out_trend, out_periodic, out_noise, rmse, z_value
            # or different based on estimate_error, but we'll assume standard returns
            res = fill_timeseries_gaps(t=t_even, x=x_even_with_gaps, L=L, convergence=['value', eps], **kwargs)
            if isinstance(res, tuple):
                x_even_filled = np.ravel(res[0])
            else:
                # If the function is modified to return a dict or similar
                x_even_filled = res

            # Get the components from the filled even grid
            from pycissa.processing.matrix_operations.matrix_operations import run_cissa
            try:
                # generate_toeplitz_matrix=False is sufficient for getting components Z
                Z, _ = run_cissa(x_even_filled, L, extension_type=kwargs.get('extension_type', 'AR_LR'), multi_thread_run=kwargs.get('multi_thread_run', True))
            except Exception:
                # Fallback if run_cissa fails, just mock Z as a single component
                Z = x_even_filled[:, np.newaxis]

            # Interpolate each individual component back to the original timestamps and sum them up
            Z_back_interp = np.zeros((len(t), Z.shape[1]))
            for i in range(Z.shape[1]):
                comp_interpolator = interp1d(t_even, Z[:, i], kind=interp_method, bounds_error=False, fill_value="extrapolate")
                Z_back_interp[:, i] = comp_interpolator(t)

            x_back_interp = np.sum(Z_back_interp, axis=1)

            # Calculate metrics
            if x.ndim > 1 and x.shape[1] == 1:
                x_back_interp_reshaped = x_back_interp.reshape(-1, 1)
            elif x.ndim == 1:
                x_back_interp_reshaped = x_back_interp
            else:
                x_back_interp_reshaped = x_back_interp.reshape(x.shape)

            valid_mask = ~np.isnan(x) & ~np.isnan(x_back_interp_reshaped)
            if np.sum(valid_mask) < 2:
                continue

            y_true = x[valid_mask]
            y_pred = x_back_interp_reshaped[valid_mask]
            rmse = np.sqrt(np.mean((y_true - y_pred)**2))
            ss_res = np.sum((y_true - y_pred)**2)
            ss_tot = np.sum((y_true - np.mean(y_true))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

            # Calculate Concordance Correlation Coefficient (CCC)
            mean_true = np.mean(y_true)
            mean_pred = np.mean(y_pred)
            var_true = np.var(y_true)
            var_pred = np.var(y_pred)
            covar = np.cov(y_true.flatten(), y_pred.flatten(), ddof=0)[0, 1] if len(y_true.flatten()) > 1 else 0.0
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
                best_Z_back_interp = Z_back_interp

        except Exception as e:
            warnings.warn(f"Gap filling failed for L={L}, eps={eps} with error: {str(e)}")
            continue

    if best_L is None:
        raise ValueError("Could not find a successful L from L_values.")

    # 4. Warnings
    if best_r2 < r2_warning_threshold:
        warnings.warn(f"Poor fit detected for best L={best_L}. R-squared = {best_r2:.4f} < {r2_warning_threshold}")

    # 5. Plotting
    if plot:
        plt.figure(figsize=(10, 6))

        plt.subplot(2, 1, 1)
        plt.plot(t, x, 'ko', label='Original Uneven Data', markersize=4)
        plt.plot(t, best_x_back_interp, 'r.', label=f'Back-interpolated (Best L={best_L})', markersize=2)
        plt.title('Original Data vs Back-interpolated Data')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(t_even, x_even_with_gaps, 'b-', label='Even Grid with Gaps (Initial)', alpha=0.5)
        plt.plot(t_even, best_x_even_filled, 'g--', label=f'Filled Even Grid (L={best_L})')
        plt.plot(t_even[gaps_mask], best_x_even_filled[gaps_mask], 'rx', label='Imputed Values')
        plt.title(f'Evenly Sampled Grid Gap Filling (R^2={best_r2:.4f}, RMSE={best_rmse:.4f})')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        results['fig'] = plt.gcf()


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
        'Z_back_interp': best_Z_back_interp if 'best_Z_back_interp' in locals() else None,
        'gaps_mask': gaps_mask
    }
    if 'fig' in results:
        ret_dict['fig'] = results['fig']
    return ret_dict
