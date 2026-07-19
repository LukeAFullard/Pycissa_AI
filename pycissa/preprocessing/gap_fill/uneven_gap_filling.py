import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.interpolate import interp1d, PchipInterpolator

def fill_uneven_timeseries(t: np.ndarray,
                           x: np.ndarray,
                           L_values: list[int],
                           dt: float,
                           gap_threshold: float,
                           interp_method: str = 'pchip',
                           optimization_metric: str = 'rmse',
                           r2_warning_threshold: float = 0.5,
                           plot: bool = True,
                           **kwargs) -> dict:
    """
    Fills gaps and centers an unevenly sampled time series by interpolating to an even grid,
    applying CiSSA spectral decomposition to reconstruct the signal, and optimizing the window length L.

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
        (Note: Kept for signature compatibility, but spectral reconstruction acts on entire grid).
    interp_method : str, optional
        Interpolation method ('linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', or 'next'). The default is 'cubic'.
    optimization_metric : str, optional
        Metric to optimize when selecting L. Either 'rmse' (minimize Root Mean Squared Error) or 'ccc' (maximize Concordance Correlation Coefficient). The default is 'rmse'.
    r2_warning_threshold : float, optional
        R-squared threshold below which a warning is issued. Default is 0.5.
    plot : bool, optional
        Whether to produce diagnostic plots.
    **kwargs :
        Additional arguments passed to run_cissa/MCissa.

    Returns
    -------
    dict
        Dictionary containing the best L, original and filled data, and statistics.
    """
    from pycissa.processing.cissa.cissa import Cissa

    # 1. Create even grid
    t_even = np.arange(np.nanmin(t), np.nanmax(t) + dt, dt)

    # Interpolate x onto t_even for initial guess
    valid = ~np.isnan(x)
    if np.sum(valid) > 1:
        if interp_method.lower() == 'pchip':
            interpolator = PchipInterpolator(t[valid], x[valid], extrapolate=True)
        else:
            interpolator = interp1d(t[valid], x[valid], kind=interp_method, bounds_error=False, fill_value="extrapolate")
        x_even_interp = interpolator(t_even)
    else:
        raise ValueError("Not enough valid data points to interpolate.")

    x_even_guess = x_even_interp.copy()

    # Apply gap_threshold masking
    idx = np.searchsorted(t, t_even)
    idx = np.clip(idx, 1, len(t) - 1)
    min_distances = np.minimum(np.abs(t_even - t[idx - 1]), np.abs(t_even - t[idx]))

    gaps_mask = min_distances > gap_threshold
    if np.any(gaps_mask):
        x_even_guess[gaps_mask] = np.nan
    else:
        warnings.warn("No gaps found based on the given gap_threshold. The entire even grid is considered known data.")

    best_L = None
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
        try:
            model = Cissa(t_even.copy(), x_even_guess.copy())

            comp_method = kwargs.get('component_selection_method', 'drop_smallest_proportion')
            prop = kwargs.get('eigenvalue_proportion', 0.95)

            if np.any(np.isnan(x_even_guess)):
                # Run iterative spectral filling
                model.pre_fill_gaps(L=L,
                                    component_selection_method=comp_method,
                                    eigenvalue_proportion=prop,
                                    **{k: v for k, v in kwargs.items() if k not in ['outliers', 'gap_threshold', 'dt', 'center_data', 'multivariate', 'estimate_error', 'verbose', 'component_selection_method', 'eigenvalue_proportion', 'alpha', '_is_fallback']})
                # After filling gaps, fit the final model
                model.fit(L=L, **{k: v for k, v in kwargs.items() if k not in ['outliers', 'gap_threshold', 'dt', 'center_data', 'multivariate', 'estimate_error', 'verbose', 'component_selection_method', 'eigenvalue_proportion', 'alpha', '_is_fallback']})
            else:
                # We run a full CiSSA spectral decomposition and reconstruction on the initial guess
                model.fit(L=L, **{k: v for k, v in kwargs.items() if k not in ['outliers', 'gap_threshold', 'dt', 'center_data', 'multivariate', 'estimate_error', 'verbose', 'component_selection_method', 'eigenvalue_proportion', 'alpha', '_is_fallback']})

            # Use grouping logic to filter the components
            try:
                # `Cissa` uses `model.Z`, not `model.Z_stacked`
                if comp_method == 'drop_smallest_proportion':
                    from pycissa.postprocessing.grouping.grouping_functions import drop_smallest_proportion_psd
                    kept_idx = drop_smallest_proportion_psd(model.Z, model.psd, prop)
                    if isinstance(kept_idx, list):
                        Z_retained = model.Z[:, kept_idx]
                    else:
                        Z_retained = kept_idx
                elif comp_method == 'monte_carlo_significant_components':
                    # Fallback for surrogate testing during iterative L optimization to avoid explosion of compute time.
                    Z_retained = model.Z
                else:
                    Z_retained = model.Z
            except Exception as e:
                warnings.warn(f"Failed to drop components: {str(e)}")
                Z_retained = model.Z # Keep all if selection fails

            # Interpolate each individual component back to the original timestamps and sum them up
            Z_back_interp = np.zeros((len(t), Z_retained.shape[1]))
            for i in range(Z_retained.shape[1]):
                if interp_method.lower() == 'pchip':
                    comp_interpolator = PchipInterpolator(t_even, Z_retained[:, i], extrapolate=True)
                else:
                    comp_interpolator = interp1d(t_even, Z_retained[:, i], kind=interp_method, bounds_error=False, fill_value="extrapolate")
                Z_back_interp[:, i] = comp_interpolator(t)

            x_back_interp = np.sum(Z_back_interp, axis=1)
            x_even_filled = np.sum(Z_retained, axis=1)
            best_Z_back_interp = Z_back_interp

            # Calculate metrics against raw measurements!
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
                best_x_even_filled = x_even_filled
                best_x_back_interp = x_back_interp
                best_Z_back_interp = Z_back_interp

        except Exception as e:
            warnings.warn(f"Spectral optimization failed for L={L} with error: {str(e)}")
            continue

    if best_L is None:
        if kwargs.get('_is_fallback', False):
            raise ValueError("Could not find a successful L from L_values in fallback mode.")
        else:
            raise ValueError("Could not find a successful L from L_values.")

    is_fallback = kwargs.pop('_is_fallback', False)

    # 4. Edge Case Fallback & Warnings
    if best_r2 < r2_warning_threshold and not is_fallback:
        try:
            if interp_method != 'linear':
                fallback_res = fill_uneven_timeseries(t=t, x=x, L_values=L_values, dt=dt, gap_threshold=gap_threshold,
                                                      interp_method='linear', optimization_metric=optimization_metric,
                                                      r2_warning_threshold=r2_warning_threshold, plot=plot, _is_fallback=True, **kwargs)
                if fallback_res['r2'] > best_r2:
                    if fallback_res['r2'] < r2_warning_threshold:
                        warnings.warn(f"Poor fit detected for best L={fallback_res['best_L']}. R-squared = {fallback_res['r2']:.4f} < {r2_warning_threshold}")
                    return fallback_res
        except Exception:
            pass

        # Second fallback for extreme sparsity: don't drop any components
        if kwargs.get('component_selection_method', 'drop_smallest_proportion') != 'none':
            try:
                kwargs_keep = kwargs.copy()
                kwargs_keep['component_selection_method'] = 'none'
                fallback_res_keep = fill_uneven_timeseries(t=t, x=x, L_values=L_values, dt=dt, gap_threshold=gap_threshold,
                                                           interp_method='linear', optimization_metric=optimization_metric,
                                                           r2_warning_threshold=r2_warning_threshold, plot=plot, _is_fallback=True, **kwargs_keep)
                if fallback_res_keep['r2'] > best_r2:
                    if fallback_res_keep['r2'] < r2_warning_threshold:
                        warnings.warn(f"Poor fit detected for best L={fallback_res_keep['best_L']}. R-squared = {fallback_res_keep['r2']:.4f} < {r2_warning_threshold}")
                    return fallback_res_keep
            except Exception:
                pass

        # If we reach here, even the fallbacks are poor (or failed). We just return the best we have, but warn.
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
        plt.plot(t_even, x_even_guess, 'b-', label='Even Grid Initial Guess', alpha=0.5)
        plt.plot(t_even, best_x_even_filled, 'g--', label=f'Spectrally Reconstructed Grid (L={best_L})')
        plt.title(f'Evenly Sampled Grid Spectral Centering (R^2={best_r2:.4f}, RMSE={best_rmse:.4f})')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        results['fig'] = plt.gcf()


    ret_dict = {
        'best_L': best_L,
        'rmse': best_rmse,
        'r2': best_r2,
        'ccc': best_ccc,
        't_even': t_even,
        'x_even_with_gaps': x_even_guess, # for backward compat
        'x_even_filled': best_x_even_filled,
        't_uneven': t,
        'x_uneven': x,
        'x_back_interp': best_x_back_interp,
        'Z_back_interp': best_Z_back_interp if 'best_Z_back_interp' in locals() else None,
        'gaps_mask': np.isnan(x_even_guess) # Mock for compat
    }
    if 'fig' in results:
        ret_dict['fig'] = results['fig']
    return ret_dict

def m_fill_uneven_timeseries(t: np.ndarray,
                             x: np.ndarray,
                             L_values: list[int],
                             dt: float,
                             gap_threshold: float,
                             interp_method: str = 'pchip',
                             optimization_metric: str = 'rmse',
                             r2_warning_threshold: float = 0.5,
                             plot: bool = True,
                             **kwargs) -> dict:
    """
    Multivariate extension of uneven gap filling and centering. Fills gaps in an unevenly sampled
    time series (T, M) by interpolating to a common even grid, applying M-CiSSA spectral decomposition
    to jointly reconstruct the signal, and optimizing the window length L.

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
        Kept for signature compatibility.
    interp_method : str, optional
        Interpolation method. Default is 'cubic'.
    optimization_metric : str, optional
        Metric to optimize when selecting L. 'rmse' or 'ccc'. Default is 'rmse'.
    r2_warning_threshold : float, optional
        R-squared threshold below which a warning is issued. Default is 0.5.
    plot : bool, optional
        Whether to produce diagnostic plots.
    **kwargs :
        Additional arguments passed to run_cissa/MCissa.

    Returns
    -------
    dict
        Dictionary containing the best L, original and filled data, and statistics.
    """
    from pycissa.processing.mcissa.mcissa import MCissa

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
            if interp_method.lower() == 'pchip':
                interpolator = PchipInterpolator(t[valid], x[valid, m], extrapolate=True)
            else:
                interpolator = interp1d(t[valid], x[valid, m], kind=interp_method, bounds_error=False, fill_value="extrapolate")
            x_even_interp[:, m] = interpolator(t_even)
        else:
            x_even_interp[:, m] = np.nan

    x_even_guess = x_even_interp.copy()

    # Apply gap_threshold masking
    idx = np.searchsorted(t, t_even)
    idx = np.clip(idx, 1, len(t) - 1)
    min_distances = np.minimum(np.abs(t_even - t[idx - 1]), np.abs(t_even - t[idx]))

    gaps_mask = min_distances > gap_threshold
    if np.any(gaps_mask):
        # Broadcast mask to (T, M)
        x_even_guess[np.tile(gaps_mask[:, np.newaxis], (1, M))] = np.nan
    else:
        warnings.warn("No gaps found based on the given gap_threshold. The entire even grid is considered known data.")

    best_L = None
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
        try:
            model = MCissa(t_even.copy(), x_even_guess.copy())

            comp_method = kwargs.get('component_selection_method', 'drop_smallest_proportion')
            prop = kwargs.get('eigenvalue_proportion', 0.95)

            if np.any(np.isnan(x_even_guess)):
                # Run iterative spectral filling
                model.pre_fill_gaps(L=L,
                                    component_selection_method=comp_method,
                                    eigenvalue_proportion=prop,
                                    multivariate=True,
                                    **{k: v for k, v in kwargs.items() if k not in ['outliers', 'gap_threshold', 'dt', 'center_data', 'multivariate', 'estimate_error', 'verbose', 'component_selection_method', 'eigenvalue_proportion', 'alpha', '_is_fallback']})
                # After filling gaps, fit the final model
                model.fit(L=L, **{k: v for k, v in kwargs.items() if k not in ['outliers', 'gap_threshold', 'dt', 'center_data', 'multivariate', 'estimate_error', 'verbose', 'component_selection_method', 'eigenvalue_proportion', 'alpha', '_is_fallback']})
            else:
                # We run a full M-CiSSA spectral decomposition and reconstruction on the initial guess
                model.fit(L=L, **{k: v for k, v in kwargs.items() if k not in ['outliers', 'gap_threshold', 'dt', 'center_data', 'multivariate', 'estimate_error', 'verbose', 'component_selection_method', 'eigenvalue_proportion', 'alpha', '_is_fallback']})

            try:
                # Manually replicate component dropping since m_select_components doesn't exist
                if comp_method == 'drop_smallest_proportion':
                    from pycissa.postprocessing.grouping.m_grouping_functions import m_classify_smallest_proportion_psd
                    trend, periodic, noise = m_classify_smallest_proportion_psd(model.Z_stacked, model.psd, L, prop)
                    # For multivariate, it returns lists of component indices!
                    kept_idx = trend + periodic
                    Z_retained = model.Z_stacked[:, :, kept_idx]
                else:
                    Z_retained = model.Z_stacked
            except Exception as e:
                warnings.warn(f"Failed to drop m components: {str(e)}")
                Z_retained = model.Z_stacked # Keep all if selection fails

            # Interpolate each individual component back to the original timestamps and sum them up
            Z_back_interp = np.zeros((len(t), M, Z_retained.shape[2]))
            for m in range(M):
                for i in range(Z_retained.shape[2]):
                    if interp_method.lower() == 'pchip':
                        comp_interpolator = PchipInterpolator(t_even, Z_retained[:, m, i], extrapolate=True)
                    else:
                        comp_interpolator = interp1d(t_even, Z_retained[:, m, i], kind=interp_method, bounds_error=False, fill_value="extrapolate")
                    Z_back_interp[:, m, i] = comp_interpolator(t)

            x_back_interp = np.sum(Z_back_interp, axis=2)
            x_even_filled = np.sum(Z_retained, axis=2)

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
                best_x_even_filled = x_even_filled
                best_x_back_interp = x_back_interp

        except Exception as e:
            warnings.warn(f"Spectral optimization failed for L={L} with error: {str(e)}")
            continue

    if best_L is None:
        if kwargs.get('_is_fallback', False):
            raise ValueError("Could not find a successful L from L_values in fallback mode.")
        else:
            raise ValueError("Could not find a successful L from L_values.")

    is_fallback = kwargs.pop('_is_fallback', False)

    # 4. Edge Case Fallback & Warnings
    if best_r2 < r2_warning_threshold and not is_fallback:
        try:
            if interp_method != 'linear':
                fallback_res = m_fill_uneven_timeseries(t=t, x=x, L_values=L_values, dt=dt, gap_threshold=gap_threshold,
                                                        interp_method='linear', optimization_metric=optimization_metric,
                                                        r2_warning_threshold=r2_warning_threshold, plot=plot, _is_fallback=True, **kwargs)
                if fallback_res['r2'] > best_r2:
                    if fallback_res['r2'] < r2_warning_threshold:
                        warnings.warn(f"Poor fit detected for best L={fallback_res['best_L']}. R-squared = {fallback_res['r2']:.4f} < {r2_warning_threshold}")
                    return fallback_res
        except Exception:
            pass

        # Second fallback for extreme sparsity: don't drop any components
        if kwargs.get('component_selection_method', 'drop_smallest_proportion') != 'none':
            try:
                kwargs_keep = kwargs.copy()
                kwargs_keep['component_selection_method'] = 'none'
                fallback_res_keep = m_fill_uneven_timeseries(t=t, x=x, L_values=L_values, dt=dt, gap_threshold=gap_threshold,
                                                             interp_method='linear', optimization_metric=optimization_metric,
                                                             r2_warning_threshold=r2_warning_threshold, plot=plot, _is_fallback=True, **kwargs_keep)
                if fallback_res_keep['r2'] > best_r2:
                    if fallback_res_keep['r2'] < r2_warning_threshold:
                        warnings.warn(f"Poor fit detected for best L={fallback_res_keep['best_L']}. R-squared = {fallback_res_keep['r2']:.4f} < {r2_warning_threshold}")
                    return fallback_res_keep
            except Exception:
                pass

        # If we reach here, even the fallbacks are poor. We just return the best we have, but warn.
        warnings.warn(f"Poor fit detected for best L={best_L}. R-squared = {best_r2:.4f} < {r2_warning_threshold}")

    # 5. Plotting
    if plot:
        fig, axes = plt.subplots(M, 1, figsize=(10, 4*M), sharex=True)
        if M == 1:
            axes = [axes]

        for m in range(M):
            axes[m].plot(t, x[:, m], 'ko', label='Original Uneven Data', markersize=4)
            axes[m].plot(t, best_x_back_interp[:, m], 'r.', label=f'Back-interpolated (L={best_L})', markersize=2)

            axes[m].plot(t_even, x_even_guess[:, m], 'b-', label='Even Grid Initial Guess', alpha=0.3)
            axes[m].plot(t_even, best_x_even_filled[:, m], 'g--', label=f'Spectrally Reconstructed Grid', alpha=0.7)

            axes[m].set_title(f'Channel {m+1}')
            axes[m].legend(fontsize='small', loc='best')
            axes[m].grid(True)

        plt.suptitle(f'Multivariate Spectral Centering (R^2={best_r2:.4f}, RMSE={best_rmse:.4f})')
        plt.tight_layout()
        results['fig'] = plt.gcf()


    ret_dict = {
        'best_L': best_L,
        'rmse': best_rmse,
        'r2': best_r2,
        'ccc': best_ccc,
        't_even': t_even,
        'x_even_with_gaps': x_even_guess,
        'x_even_filled': best_x_even_filled,
        't_uneven': t,
        'x_uneven': x,
        'x_back_interp': best_x_back_interp,
    }
    if 'fig' in results:
        ret_dict['fig'] = results['fig']
    return ret_dict
