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
    valid = ~np.isnan(x)
    if np.sum(valid) > 1:
        interpolator = interp1d(t[valid], x[valid], kind=interp_method, bounds_error=False, fill_value="extrapolate")
        x_even_interp = interpolator(t_even)
    else:
        x_even_interp = np.full_like(t_even, np.nan)

    # 2. Identify gaps
    # For each point in t_even, find the distance to the closest point in t
    # If distance > gap_threshold, set x_even_interp to NaN
    # We use searchsorted for efficient nearest neighbor distance calculation
    idx = np.searchsorted(t, t_even)
    idx = np.clip(idx, 1, len(t) - 1)
    left_dist = np.abs(t_even - t[idx - 1])
    right_dist = np.abs(t_even - t[idx])
    min_distances = np.minimum(left_dist, right_dist)

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
            valid_mask = ~np.isnan(x) & ~np.isnan(x_back_interp)
            if np.sum(valid_mask) < 2:
                continue

            y_true = x[valid_mask]
            y_pred = x_back_interp[valid_mask]
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
    Multivariate extension of uneven gap filling. Fills gaps in an unevenly sampled
    time series (T, M) by interpolating to a common even grid, applying joint M-CiSSA
    gap filling while optimizing the window length L, and projecting back.

    Parameters
    ----------
    t : np.ndarray
        Unevenly sampled time points (T,).
    x : np.ndarray
        Values at uneven time points (T, M).
    L_values : list[int]
        List of L (window length) parameters to optimize over for CiSSA gap filling.
    dt : float
        Grid spacing for the evenly sampled grid.
    gap_threshold : float
        Max distance to a real data point on the even grid before it is considered a gap (NaN).
    eps_values : list[float], optional
        List of convergence epsilon values to optimize over. If None, defaults to [1.0].
    interp_method : str, optional
        Interpolation method. Default is 'cubic'.
    optimization_metric : str, optional
        Metric to optimize when selecting L. 'rmse' or 'ccc'. Default is 'rmse'.
    r2_warning_threshold : float, optional
        R-squared threshold below which a warning is issued. Default is 0.5.
    plot : bool, optional
        Whether to produce diagnostic plots.
    **kwargs :
        Additional arguments passed to m_fill_timeseries_gaps.

    Returns
    -------
    dict
        Dictionary containing the best L, original and filled data, and statistics.
    """
    from pycissa.preprocessing.gap_fill.gap_filling import m_fill_timeseries_gaps

    if x.ndim != 2:
        raise ValueError("x must be a 2D array of shape (T, M) for multivariate filling.")

    T, M = x.shape

    # 1. Create even grid
    t_even = np.arange(np.nanmin(t), np.nanmax(t) + dt, dt)

    # Interpolate x onto t_even for initial guess
    x_even_interp = np.zeros((len(t_even), M))
    for m in range(M):
        valid = ~np.isnan(x[:, m])
        if np.sum(valid) > 1:
            interpolator = interp1d(t[valid], x[valid, m], kind=interp_method, bounds_error=False, fill_value="extrapolate")
            x_even_interp[:, m] = interpolator(t_even)
        else:
            x_even_interp[:, m] = np.nan

    # 2. Identify gaps
    idx = np.searchsorted(t, t_even)
    idx = np.clip(idx, 1, len(t) - 1)
    left_dist = np.abs(t_even - t[idx - 1])
    right_dist = np.abs(t_even - t[idx])
    min_distances = np.minimum(left_dist, right_dist)

    x_even_with_gaps = x_even_interp.copy()
    gaps_mask_1d = min_distances > gap_threshold

    # Broadcast gaps_mask_1d across all channels.
    for m in range(M):
        x_even_with_gaps[gaps_mask_1d, m] = np.nan

        # Also ensure original nans propogate if interp1d hid them
        # If a channel has NaNs, we need to map those NaNs onto the even grid to trigger gap filling!
        nan_indices = np.where(np.isnan(x[:, m]))[0]
        for nan_idx in nan_indices:
            # Find the closest even grid point to the original timestamp that was NaN
            closest_even_idx = np.argmin(np.abs(t_even - t[nan_idx]))
            # If the even point is close enough to the NaN timestamp, it should also be NaN
            if np.abs(t_even[closest_even_idx] - t[nan_idx]) <= gap_threshold:
                x_even_with_gaps[closest_even_idx, m] = np.nan

    gaps_mask = np.isnan(x_even_with_gaps)

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
            # m_fill_timeseries_gaps returns: x_ca, out, out_trend, out_periodic, out_noise, rmse, z_value
            res = m_fill_timeseries_gaps(t=t_even, x=x_even_with_gaps, L=L, convergence=['value', eps], **kwargs)
            if isinstance(res, tuple):
                x_even_filled = res[0]
            else:
                x_even_filled = res

            # Get the components from the filled even grid
            from pycissa.processing.mcissa.mcissa import MCissa
            try:
                # Use MCissa to get components
                model = MCissa(t_even, x_even_filled)
                model.fit(L=L)
                Z = model.Z_stacked # Shape: (T_even, M, nft)
            except Exception:
                Z = x_even_filled[:, :, np.newaxis]

            # Interpolate each individual component back to the original timestamps and sum them up
            Z_back_interp = np.zeros((len(t), M, Z.shape[2]))
            for m in range(M):
                for i in range(Z.shape[2]):
                    comp_interpolator = interp1d(t_even, Z[:, m, i], kind=interp_method, bounds_error=False, fill_value="extrapolate")
                    Z_back_interp[:, m, i] = comp_interpolator(t)

            x_back_interp = np.sum(Z_back_interp, axis=2)

            # Calculate metrics using valid mask flattened
            valid_mask = ~np.isnan(x) & ~np.isnan(x_back_interp)
            if np.sum(valid_mask) < 2:
                continue

            y_true = x[valid_mask].flatten()
            y_pred = x_back_interp[valid_mask].flatten()

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
        fig, axes = plt.subplots(M, 1, figsize=(10, 4*M), sharex=True)
        if M == 1:
            axes = [axes]

        for m in range(M):
            axes[m].plot(t, x[:, m], 'ko', label='Original Uneven Data', markersize=4)
            axes[m].plot(t, best_x_back_interp[:, m], 'r.', label=f'Back-interpolated (L={best_L})', markersize=2)

            axes[m].plot(t_even, x_even_with_gaps[:, m], 'b-', label='Even Grid Gaps', alpha=0.3)
            axes[m].plot(t_even, best_x_even_filled[:, m], 'g--', label=f'Filled Even Grid', alpha=0.7)

            gaps = gaps_mask[:, m]
            axes[m].plot(t_even[gaps], best_x_even_filled[gaps, m], 'rx', label='Imputed Values')

            axes[m].set_title(f'Channel {m+1}')
            axes[m].legend(fontsize='small', loc='best')
            axes[m].grid(True)

        plt.suptitle(f'Multivariate Uneven Gap Filling (R^2={best_r2:.4f}, RMSE={best_rmse:.4f})')
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
