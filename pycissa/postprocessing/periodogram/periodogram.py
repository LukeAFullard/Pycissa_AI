import numpy as np
import matplotlib.pyplot as plt
import piecewise_regression
import statsmodels.api as sm
import warnings
import nolds
min_width = 720
min_height = 570
###############################################################################
###############################################################################

def make_periodogram_arrays(psd                    : np.array,
                            frequencies            : dict,
                            significant_components :  list|None = None,
                            ) -> tuple[list,list]:
    '''
    Function to arrange the psd array and frequency dictionary from CiSSA into two lists ready for plotting.

    Parameters
    ----------
    psd : np.array
        DESCRIPTION. Power Spectral Density array.
    frequencies : dict
        DESCRIPTION. Dictionary of frequencies (per unit time-step) and array location.
    significant_components : list|None, optional
        DESCRIPTION. The default is None. If provided, significant_components is a list of integers. These components will be ignored in the  list of frequencies.

    Returns
    -------
    my_freq,my_psd  : tuple[list,list]
        DESCRIPTION. list of frequencies and psd.

    '''
    if significant_components is None:
        significant_components = []

    psd = psd.reshape(len(psd),)
    reverse_dictionary  = {value[0]: key for key, value in frequencies.items()}
    reverse_dictionary = dict(sorted(reverse_dictionary.items()))

    removed_psd  = []
    removed_freq = []
    my_psd  = []
    my_freq = []
    for key_j in reverse_dictionary.keys():
        if not key_j in significant_components:
            if not reverse_dictionary.get(key_j) == 'trend':
                my_psd.append(psd[key_j])
                my_freq.append(reverse_dictionary.get(key_j))
        else:
            if not reverse_dictionary.get(key_j) == 'trend':
                removed_psd.append(psd[key_j])
                removed_freq.append(reverse_dictionary.get(key_j))
    return my_freq,my_psd,removed_psd,removed_freq

###############################################################################
###############################################################################
def lomb_scargle_astropy(x             : np.ndarray,
                         my_freq       : list,
                         normalization : str  = 'standard',
                         center_data   : bool = True,
                         fit_mean      : bool = True,
                         nterms        : int  = 1):
    '''
    Function to fit a Lomb-Scargle (normalised by default) periodogram. Note frequencies are cycles per unit time.
    By default we fit the periodogram to the same frequencies as CiSSA for an easy comparison.
    See https://astropy-cjhang.readthedocs.io/en/latest/stats/lombscargle.html.

    Price-Whelan, A. M., Lim, P. L., Earl, N., Starkman, N., Bradley, L., Shupe, D. L., ... & Astropy Collaboration. (2022). The astropy project: sustaining and growing a community-oriented open-source project and the latest major release (v5. 0) of the core package. The Astrophysical Journal, 935(2), 167.

    Parameters
    ----------
    x : np.ndarray
        DESCRIPTION. Input data array.
    my_freq : list
        DESCRIPTION. Frequencies to fit the periodogram.
    normalization : str, optional
        DESCRIPTION. The default is 'standard'. How to normalise the periodogram. See https://astropy-cjhang.readthedocs.io/en/latest/stats/lombscargle.html.
    center_data : bool, optional
        DESCRIPTION. The default is True. Center data? See https://astropy-cjhang.readthedocs.io/en/latest/stats/lombscargle.html.
    fit_mean : bool, optional
        DESCRIPTION. The default is True. Fit mean? See https://astropy-cjhang.readthedocs.io/en/latest/stats/lombscargle.html.
    nterms : int, optional
        DESCRIPTION. The default is 1. Number of terms in the periodogram.

    Returns
    -------
    power : TYPE
        DESCRIPTION. Lomb-Scargle power.

    '''
    try:
        from astropy.timeseries import LombScargle
    except ImportError:
        from astropy.stats import LombScargle
    #NOTE: In astropy, all frequencies in LombScargle are not angular frequencies, but rather frequencies of oscillation; i.e. number of cycles per unit time.
    power = LombScargle(np.linspace(1,len(x),len(x)),
                        x,
                        normalization=normalization,
                        center_data=center_data,
                        fit_mean=fit_mean,
                        nterms=nterms,
                        ).power(my_freq)
    return power


#
def linear_fit(my_freq : list,
               my_psd  : list,
               alpha   : float=0.05,) -> dict:
    '''
    Function to perform OLS fitting to the log10(psd) vs log10(frequency).

    Parameters
    ----------
    my_freq : list
        DESCRIPTION. List of frequencies.
    my_psd : list
        DESCRIPTION. List of psd.
    alpha : float, optional
        DESCRIPTION. The default is 0.05. Significance level for calculating confidence interval (CI = 100*(1-alpha)).
    Returns
    -------
    ols_result : dict
        DESCRIPTION. Dictionary of results.

    '''
    Z = np.array([np.log10(my_freq)])
    Z = Z.T
    Z = sm.add_constant(Z, has_constant='add')
    # Basic OLS fit
    results = sm.OLS(endog=np.array(np.log10(my_psd)), exog=Z).fit()

    ols_result = {
        'constant'  :  {
                        'result'              : results.params[0],
                        'confidence_interval' : results.conf_int(alpha=alpha)[0],
                        },
        'slope'     :  {'result'              : results.params[1],
                        'confidence_interval' : results.conf_int(alpha=alpha)[1],
                        },
        }
    return ols_result
###############################################################################
###############################################################################
def robust_segmented_fit(my_freq         : list,
                         my_psd          : list,
                         break_point     : float,
                         alpha           : float=0.05,) -> dict:
    '''
    Function to reanalyse the segmented result but using robust regression.
    Takes the break_point from standard segmented regression then uses robust regression on the left and then right sides of this point.

    Parameters
    ----------
    my_freq : list
        DESCRIPTION. List of frequencies
    my_psd : list
        DESCRIPTION. List of psd.
    break_point : float
        DESCRIPTION. Frequency of the breakpoint in segmented fit.
    alpha : float, optional
        DESCRIPTION. The default is 0.05. Significance level for calculating confidence interval (CI = 100*(1-alpha)).

    Returns
    -------
    robust_segmented_result : dict
        DESCRIPTION. Dictionary of results.

    '''
    robust_slopes = {}
    #deal with lower frequency
    lower_freq = [x for x in my_freq if x<=break_point]
    lower_psd  = [x for x,y in zip(my_psd,my_freq) if y <= break_point]
    if not len(lower_freq) > 3: #when fitting we want at least 3 points for a linear fit
        return {}
    #deal with upper frequency
    upper_freq = [x for x in my_freq if x>=break_point]
    upper_psd  = [x for x,y in zip(my_psd,my_freq) if y >= break_point]
    if not len(upper_freq) > 3: #when fitting we want at least 3 points for a linear fit
        return {}

    lower_result = robust_linear_fit(lower_freq,lower_psd,alpha)
    upper_result = robust_linear_fit(upper_freq,upper_psd,alpha)


    robust_segmented_result = {
        'lower_freq'                    : lower_freq,
        'upper_freq'                    : upper_freq,
        'breakpoint'                    : break_point,
        'slope_less_than_breakpoint'    : {'constant'             : lower_result.get('constant').get('result'),
                                           'slope'                : lower_result.get('slope').get('result'),
                                           'confidence_interval'  : lower_result.get('slope').get('confidence_interval')},
        'slope_greater_than_breakpoint' : {'constant'             : upper_result.get('constant').get('result'),
                                           'slope'                : upper_result.get('slope').get('result'),
                                           'confidence_interval'  : upper_result.get('slope').get('confidence_interval')},

        }
    return robust_segmented_result
###############################################################################
###############################################################################
def plot_robust_segmented_linear_fit(my_freq                  : list,
                                    my_psd                    : list,
                                    alpha                     : float,
                                    removed_psd               : list,
                                    removed_freq              : list,
                                    robust_segmented_result   : dict,
                                    legend_label              : str = 'Robust segmented fit',
                                    title                     : str = 'Periodogram - robust segmented fit',
                                    **kwargs):
        '''
        Function to plot a scatter plot of the data + the linear fit.

        Parameters
        ----------
        my_freq : list
            DESCRIPTION. List of frequencies.
        my_psd : list
            DESCRIPTION. List of psd.
        alpha : float
            DESCRIPTION. Significance level for calculating confidence interval (CI = 100*(1-alpha)).
        robust_segmented_result : dict
            DESCRIPTION. Dictionary of fit results.
        legend_label : str, optional
            DESCRIPTION. The default is 'linear fit'. Legend label text.
        title : str, optional
            DESCRIPTION. The default is 'Periodogram - linear fit'. Title text.
        **kwargs : TYPE
            DESCRIPTION. Plotting kwargs to pass to matplotlib.

        Returns
        -------
        fig : figure
            DESCRIPTION. Plot of log10(frequency) vs log10(psd) with a linear fit.

        '''
        fig, ax = plt.subplots(figsize=(8, 6))

        #scatter data
        ax.scatter(np.log10(my_freq), np.log10(my_psd),color='k',label='data', **kwargs)
        ax.scatter(np.log10(removed_freq), np.log10(removed_psd),color='orange',label='removed frequencies', **kwargs)

        #plot break point
        ax.axvline(np.log10(robust_segmented_result.get('breakpoint')), **kwargs)

        #axes labels etc
        ax.set_ylabel('log(psd)')
        ax.set_xlabel('log(frequency) (cycles per timestep)')
        ax.legend(loc='upper right')
        fig.suptitle('Periodogram - robust segmented linear fit', y=1.15, fontsize=18)

        #add line:
        lower_freq = robust_segmented_result.get('lower_freq')
        xx_plot = np.linspace(min(np.log10(lower_freq)), np.log10(robust_segmented_result.get('breakpoint')), 100)
        yy_plot = robust_segmented_result.get('slope_less_than_breakpoint').get('constant') + xx_plot*robust_segmented_result.get('slope_less_than_breakpoint').get('slope')
        ax.plot(xx_plot, yy_plot,'r',label=legend_label, **kwargs)

        upper_freq = robust_segmented_result.get('upper_freq')
        xx_plot = np.linspace(np.log10(robust_segmented_result.get('breakpoint')), max(np.log10(upper_freq)), 100)
        yy_plot = robust_segmented_result.get('slope_greater_than_breakpoint').get('constant') + xx_plot*robust_segmented_result.get('slope_greater_than_breakpoint').get('slope')
        ax.plot(xx_plot, yy_plot,'r', **kwargs)


        # Extract and format values
        breakpoint1_estimate = robust_segmented_result.get('breakpoint')
        alpha1_estimate = robust_segmented_result.get('slope_less_than_breakpoint').get('slope')
        alpha1_ci = robust_segmented_result.get('slope_less_than_breakpoint').get('confidence_interval')
        alpha1_ci_lower, alpha1_ci_upper = alpha1_ci  # Unpack tuple

        alpha2_estimate = robust_segmented_result.get('slope_greater_than_breakpoint').get('slope')
        alpha2_ci = robust_segmented_result.get('slope_greater_than_breakpoint').get('confidence_interval')
        alpha2_ci_lower, alpha2_ci_upper = alpha2_ci  # Unpack tuple

        # Set title with formatted values
        title_text = (
            f"Slopes with {int(100 * (1 - alpha))}% confidence interval.\n"
            f"For frequency < {breakpoint1_estimate:.4f}, slope: {alpha1_estimate:.3f} "
            f"({alpha1_ci_lower:.3f} - {alpha1_ci_upper:.3f})\n"
            f"For frequency > {breakpoint1_estimate:.4f}, slope: {alpha2_estimate:.3f} "
            f"({alpha2_ci_lower:.3f} - {alpha2_ci_upper:.3f})"
        )

        ax.set_title(title_text, fontsize=10)
        # Get the current figure size in inches and DPI
        fig_width_inch, fig_height_inch = fig.get_size_inches()
        dpi = fig.get_dpi()

        # Convert to pixels
        width_px = fig_width_inch * dpi
        height_px = fig_height_inch * dpi

        if width_px < min_width or height_px < min_height:
            new_width_inch = max(min_width / dpi, fig_width_inch)
            new_height_inch = max(min_height / dpi, fig_height_inch)
            fig.set_size_inches(new_width_inch, new_height_inch)
        return fig
###############################################################################
###############################################################################
def robust_linear_fit(my_freq : list,
                      my_psd  : list,
                      alpha   : float=0.05,
                      ) -> dict:
    '''
    Function to perform ROBUST linear fitting to the log10(psd) vs log10(frequency).

    Parameters
    ----------
    my_freq : list
        DESCRIPTION. List of frequencies.
    my_psd : list
        DESCRIPTION. List of psd.
    alpha : float, optional
        DESCRIPTION. The default is 0.05. Significance level for calculating confidence interval (CI = 100*(1-alpha)).
    Returns
    -------
    robust_result : dict
        DESCRIPTION. Dictionary of results.

    '''
    from statsmodels.robust.scale import HuberScale
    # Convert frequency and PSD to log scale
    X = np.log10(my_freq)
    Y = np.log10(my_psd)
    # Add a constant term for the intercept
    X = sm.add_constant(X)

    # Fit a robust linear model using Huber's T norm
    model = sm.RLM(Y, X, M=sm.robust.norms.HuberT())
    # model = sm.RLM(Y, X, M=sm.robust.norms.MQuantileNorm(0.5,sm.robust.norms.LeastSquares()))
    # model = sm.RLM(Y, X, M=sm.robust.norms.MQuantileNorm(0.5,sm.robust.norms.HuberT()))
    # model = sm.RLM(Y, X, M=sm.robust.norms.LeastSquares())
    results = model.fit(scale_est=HuberScale())

    # Extract results
    robust_result = {
        'constant': {
            'result': results.params[0],
            'confidence_interval': results.conf_int(alpha=alpha)[0],
        },
        'slope': {
            'result': results.params[1],
            'confidence_interval': results.conf_int(alpha=alpha)[1],
        },

    }

    return robust_result

###############################################################################
###############################################################################
def segmented_regression(my_freq         :  list,
                         my_psd          :  list,
                         max_breakpoints :  int = 1,
                         n_boot          :  int = 500
                         )-> tuple[list,list]:
    '''
    Function to perform segmented regression (multiple linear fits) to the log10(psd) vs log10(frequency).
    See:
    Pilgrim, C. (2021). piecewise-regression (aka segmented regression) in Python. Journal of Open Source Software, 6(68).
    https://joss.theoj.org/papers/10.21105/joss.03859
    https://github.com/chasmani/piecewise-regression

    Parameters
    ----------
    my_freq : list
        DESCRIPTION. List of frequencies.
    my_psd : list
        DESCRIPTION. List of psd.
    max_breakpoints : int, optional
        DESCRIPTION. The default is 1. Max number of linear breakpoints to consider.
    n_boot : int, optional
        DESCRIPTION. The default is 500. Number of bootstraps for calclating confidence intervals on the breakpoint location.

    Returns
    -------
    model_summaries : list
        DESCRIPTION. List of model summaries
    models : list
        DESCRIPTION. List of model result class.

    '''


    if max_breakpoints > 1:
        max_breakpoints = 1
        warnings.warn("For now max_breakpoints must be 0 or 1. This may change in the future. Resetting value to 1,")
    ms = piecewise_regression.ModelSelection(np.log10(my_freq),
                                             np.log10(my_psd),
                                             max_breakpoints=max_breakpoints,n_boot=n_boot,
                                             verbose=False)

    model_summaries = [x for x in ms.model_summaries if x['converged'] == True and x['n_breakpoints'] == 1]
    models = [x for x in ms.models if x.best_muggeo and x.n_breakpoints == 1]
    return model_summaries,models

###############################################################################
###############################################################################
def plot_linear_fit(my_freq      : list,
                    my_psd       : list,
                    alpha        : float,
                    ols_result   : dict,
                    removed_psd  : list,
                    removed_freq : list,
                    legend_label : str = 'linear fit',
                    title        : str = 'Periodogram - linear fit',
                    **kwargs):
    '''
    Function to plot a scatter plot of the data + the linear fit.

    Parameters
    ----------
    my_freq : list
        DESCRIPTION. List of frequencies.
    my_psd : list
        DESCRIPTION. List of psd.
    alpha : float
        DESCRIPTION. Significance level for calculating confidence interval (CI = 100*(1-alpha)).
    ols_result : dict
        DESCRIPTION. Dictionary of fit results.
    legend_label : str, optional
        DESCRIPTION. The default is 'linear fit'. Legend label text.
    title : str, optional
        DESCRIPTION. The default is 'Periodogram - linear fit'. Title text.
    **kwargs : TYPE
        DESCRIPTION. Plotting kwargs to pass to matplotlib.

    Returns
    -------
    fig : figure
        DESCRIPTION. Plot of log10(frequency) vs log10(psd) with a linear fit.

    '''
    fig, ax = plt.subplots(figsize=(8, 6))

    #scatter data
    ax.scatter(np.log10(my_freq), np.log10(my_psd),color='k',label='data', **kwargs)
    ax.scatter(np.log10(removed_freq), np.log10(removed_psd),color='orange',label='removed frequencies', **kwargs)

    #add line:
    xx_plot = np.linspace(min(np.log10(my_freq)), max(np.log10(my_freq)), 100)
    yy_plot = ols_result.get('constant').get('result') + xx_plot*ols_result.get('slope').get('result')
    ax.plot(xx_plot, yy_plot,'r',label=legend_label, **kwargs)

    #axes labels etc
    ax.set_ylabel('log(psd)')
    ax.set_xlabel('log(frequency) (cycles per timestep)')
    ax.legend(loc='upper right')
    fig.suptitle(title, y=1.15, fontsize=18)
    constant_result = f"{ols_result['constant']['result']:.3f}"
    constant_ci = [f"{ci:.3f}" for ci in ols_result['constant']['confidence_interval']]
    slope_result = f"{ols_result['slope']['result']:.3f}"
    slope_ci = [f"{ci:.3f}" for ci in ols_result['slope']['confidence_interval']]
    ax.set_title(
    f"Fitting parameters with {int(100*(1-alpha))}% confidence interval.\n"
    f"Constant: {constant_result} "
    f"({constant_ci[0]} - {constant_ci[1]})\n"
    f"Slope: {slope_result} "
    f"({slope_ci[0]} - {slope_ci[1]})",
    fontsize=10
    )
    # Get the current figure size in inches and DPI
    fig_width_inch, fig_height_inch = fig.get_size_inches()
    dpi = fig.get_dpi()

    # Convert to pixels
    width_px = fig_width_inch * dpi
    height_px = fig_height_inch * dpi

    if width_px < min_width or height_px < min_height:
        new_width_inch = max(min_width / dpi, fig_width_inch)
        new_height_inch = max(min_height / dpi, fig_height_inch)
        fig.set_size_inches(new_width_inch, new_height_inch)
    return fig

###############################################################################
###############################################################################
def plot_segmented_fit(my_freq         : list,
                       my_psd          : list,
                       alpha           : float,
                       removed_psd     : list,
                       removed_freq    : list,
                       model_summaries : list,
                       models          : list,
                       **kwargs):
    '''
    Function to plot a scatter plot of the data + the segmented linear fit.

    Parameters
    ----------
    my_freq : list
        DESCRIPTION. List of frequencies.
    my_psd : list
        DESCRIPTION. List of psd.
    alpha : float
        DESCRIPTION. Significance level for calculating confidence interval (CI = 100*(1-alpha)).
    model_summaries : list
        DESCRIPTION. List of model summary results.
    models : list
        DESCRIPTION. List of full model fit results.
    **kwargs : TYPE
        DESCRIPTION. Plotting kwargs to pass to matplotlib.

    Returns
    -------
    fig : TYPE
        DESCRIPTION. Plot of log10(frequency) vs log10(psd) with a segmented linear fit.

    '''
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_summary_j in model_summaries:
        #only using 1 breakpoint plot. May change this in the future.
        if model_summary_j.get('n_breakpoints') == 1:
            #scatter data
            ax.scatter(np.log10(my_freq), np.log10(my_psd),color='k',label='data', **kwargs)
            ax.scatter(np.log10(removed_freq), np.log10(removed_psd),color='orange',label='removed frequencies', **kwargs)

            #plot lines
            for model_k in models:
                if model_k.n_breakpoints == 1:
                    xx_plot = np.linspace(min(np.log10(my_freq)), max(np.log10(my_freq)), 100)
                    yy_plot = model_k.predict(xx_plot)
                    ax.plot(xx_plot, yy_plot,'r',label='segmented linear fit', **kwargs)
                    #axes labels etc
                    ax.set_ylabel('log(psd)')
                    ax.set_xlabel('log(frequency) (cycles per timestep)')
                    ax.legend(loc='upper right')
                    fig.suptitle('Periodogram - segmented linear fit', y=1.15, fontsize=18)

                    #plot breakpoint
                    breakpoints = model_k.best_muggeo.best_fit.next_breakpoints
                    for bp in breakpoints:
                        ax.axvline(bp, **kwargs)

                    # Extract and format values
                    breakpoint1_estimate = np.power(10, model_summary_j['estimates']['breakpoint1']['estimate'])
                    alpha1_estimate = model_summary_j['estimates']['alpha1']['estimate']
                    alpha1_ci = model_summary_j['estimates']['alpha1']['confidence_interval']
                    alpha1_ci_lower, alpha1_ci_upper = alpha1_ci  # Unpack tuple

                    alpha2_estimate = model_summary_j['estimates']['alpha2']['estimate']
                    alpha2_ci = model_summary_j['estimates']['alpha2']['confidence_interval']
                    alpha2_ci_lower, alpha2_ci_upper = alpha2_ci  # Unpack tuple

                    # Set title with formatted values
                    title_text = (
                        f"Slopes with {int(100 * (1 - alpha))}% confidence interval.\n"
                        f"For frequency < {breakpoint1_estimate:.4f}, slope: {alpha1_estimate:.3f} "
                        f"({alpha1_ci_lower:.3f} - {alpha1_ci_upper:.3f})\n"
                        f"For frequency > {breakpoint1_estimate:.4f}, slope: {alpha2_estimate:.3f} "
                        f"({alpha2_ci_lower:.3f} - {alpha2_ci_upper:.3f})"
                    )

                    ax.set_title(title_text, fontsize=10)
    # Get the current figure size in inches and DPI
    fig_width_inch, fig_height_inch = fig.get_size_inches()
    dpi = fig.get_dpi()

    # Convert to pixels
    width_px = fig_width_inch * dpi
    height_px = fig_height_inch * dpi

    if width_px < min_width or height_px < min_height:
        new_width_inch = max(min_width / dpi, fig_width_inch)
        new_height_inch = max(min_height / dpi, fig_height_inch)
        fig.set_size_inches(new_width_inch, new_height_inch)
    return fig

###############################################################################
###############################################################################
def plot_rolling_hurst(rolling_hurst,rolling_hurst_detrended,window):
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot for the top subplot
    axs[0].plot(rolling_hurst, 'k')

    # Set labels and title for the top subplot
    axs[0].set_ylabel('Hurst exponent')
    axs[0].set_title(f'Rolling Hurst Exponent of original series - window = {window} time steps')

    # Plot for the top subplot
    axs[1].plot(rolling_hurst_detrended, 'r')

    # Set labels and title for the top subplot
    axs[1].set_ylabel('Hurst exponent')
    axs[1].set_title(f'Rolling Hurst Exponent of detrended series - window = {window} time steps')

    # Adjust layout
    plt.tight_layout()
    # Get the current figure size in inches and DPI
    fig_width_inch, fig_height_inch = fig.get_size_inches()
    dpi = fig.get_dpi()

    # Convert to pixels
    width_px = fig_width_inch * dpi
    height_px = fig_height_inch * dpi

    if width_px < min_width or height_px < min_height:
        new_width_inch = max(min_width / dpi, fig_width_inch)
        new_height_inch = max(min_height / dpi, fig_height_inch)
        fig.set_size_inches(new_width_inch, new_height_inch)
    # Return the figure object
    return fig

###############################################################################
###############################################################################
def calculate_hurst_exponent(x_trend     : np.ndarray,
                             x_detrended : np.ndarray,
                             **kwargs) -> tuple[float,float]:
    '''
    Function to calculate Hurst exponent using the nolds package, https://cschoel.github.io/nolds/

    Parameters
    ----------
    x_trend : np.ndarray
        DESCRIPTION. Array of trend of series.
    x_detrended : np.ndarray
        DESCRIPTION. Array of detrended series.
    **kwargs : TYPE
        DESCRIPTION. kwargs for nolds.

    Returns
    -------
    all_hurst,detrended_hurst : tuple[float,float]
        DESCRIPTION. Hurst exponent for the full signal and the detrended series.

    '''

    all_hurst = nolds.hurst_rs(x_trend+x_detrended,fit = 'poly', **kwargs)
    detrended_hurst = nolds.hurst_rs(x_detrended,fit = 'poly', **kwargs)
    return all_hurst,detrended_hurst

def calculate_rolling_hurst_exponent(x_trend: np.ndarray,
                                     x_detrended: np.ndarray,
                                     window: int,
                                     **kwargs) -> tuple[np.ndarray,np.ndarray]:
    '''
    Calculate the rolling Hurst exponent of a time series without using pandas.

    Parameters
    ----------
    x_trend : np.ndarray
        DESCRIPTION. numpy array of the trend component of the time series.
    x_detrended : np.ndarray
        DESCRIPTION. numpy array of the detrended component of the time series.
    window : int
        DESCRIPTION. size of the rolling window.
    **kwargs : TYPE
        DESCRIPTION. additional arguments to pass to nolds.hurst_rs.

    Returns
    -------
    rolling_hurst : np.ndarray
        DESCRIPTION. Rolling Hurst exponent for the full time series.
    rolling_hurst_detrended : np.ndarray
        DESCRIPTION. Rolling Hurst exponent for the detrended time series.

    '''
    # Combine the trend and detrended components
    combined_series = x_trend + x_detrended
    n = len(combined_series)

    # Initialize an array to store the rolling Hurst exponent values
    rolling_hurst = np.full(n - window + 1, np.nan)
    rolling_hurst_detrended = np.full(n - window + 1, np.nan)

    # Compute Hurst exponent for each window
    for i in range(n - window + 1):
        window_series = combined_series[i:i + window]
        rolling_hurst[i] = nolds.hurst_rs(window_series,fit = 'poly', **kwargs)

        window_series = x_detrended[i:i + window]
        rolling_hurst_detrended[i] = nolds.hurst_rs(window_series,fit = 'poly', **kwargs)

    return rolling_hurst,rolling_hurst_detrended
###############################################################################
###############################################################################
def generate_peridogram_plots(
                            x_trend                : np.ndarray,
                            x_detrended            : np.ndarray,
                            psd                    : np.array,
                            frequencies            : dict,
                            significant_components : list|None = None,
                            alpha                  : float = 0.05,
                            max_breakpoints        : int = 1,
                            n_boot                 : int = 500,
                            hurst_window           : int = 12,
                            **kwargs):
    '''
    Function to run a periodogram analysis to find the fractal scaling of the time series.
    In all cases the trend is not considered, and significant periodic components can be ignored too using the input parameters.
    Also calculates the Hurst exponent for the full and detrended series.

    Parameters
    ----------
    x_trend : np.ndarray
        DESCRIPTION. numpy array of the trend component of the time series.
    x_detrended : np.ndarray
        DESCRIPTION. numpy array of the detrended component of the time series.
    psd : np.array
        DESCRIPTION. Power specrtal density from the CiSSA fit.
    frequencies : dict
        DESCRIPTION. Frequencies from the Cissa fit.
    significant_components : list|None, optional
        DESCRIPTION. The default is None. The default is None. A list of significant components which will not be considered in the periodogram analysis. Can also be None, in which case all components (except the trend) will be used for the periodogram, or if significant_components = None and monte_carlo_significant_components = True, then the monte carlo significant components will be removed.
    alpha : float, optional
        DESCRIPTION. The default is 0.05. Significance level for statistical tests.
    max_breakpoints : int, optional
        DESCRIPTION. The default is 1. Max number of breakpoints for the segmented linear fit. Currently will always be reset to 1 if >1.
    n_boot : int, optional
        DESCRIPTION. The default is 500. Number of bootstrap iterations for the segmented linear fit.
    hurst_window : int, optional
        DESCRIPTION. The default is 12. The window length (in number of time steps) for the rolling Hurst calculation.
    **kwargs : TYPE
        DESCRIPTION. keyword arguments for segmented fitting.

    Returns
    -------
    fig_linear : figure
        DESCRIPTION. Linear fit of log10(frequencies) vs log10(psd).
    fig_segmented : figure
        DESCRIPTION. Segmented linear fit of log10(frequencies) vs log10(psd).
    fig_robust_linear : figure
        DESCRIPTION. Robust linear fit of log10(frequencies) vs log10(psd).
    linear_slopes : dict
        DESCRIPTION. Dictionary of results from linear fitting.
    segmented_slopes : dict
        DESCRIPTION. Dictionary of results from segmented linear fitting.
    robust_linear_slopes : dict
        DESCRIPTION. Dictionary of results from robust linear fitting.
    all_hurst : float
        DESCRIPTION. Hurst exponent of the full time series.
    detrended_hurst : float
        DESCRIPTION. Hurst exponent of the detrended time series.
    fig_rolling_hurst : figure
        DESCRIPTION. Figure of the rolling Hurst exponent (unsure if this is correct, use with caution).
    rolling_hurst : np.ndarray
        DESCRIPTION. Array of rolling Hurst exponents for the full time series (unsure if this is correct, use with caution).
    rolling_hurst_detrended : np.ndarray
        DESCRIPTION. Array of rolling Hurst exponents for the detrended time series (unsure if this is correct, use with caution).

    '''

    y = x_trend + x_detrended

    #get psd and frequencies of interest
    my_freq,my_psd_,removed_psd,removed_freq = make_periodogram_arrays(psd, frequencies,significant_components=significant_components)
    if not len(my_freq) > 3:  #when fitting we want at least 3 points for a linear fit
        return None, None, None, {}, {}, None,None,None, None,None,None,None,None
    mean_y_sq = np.mean(y**2)
    if mean_y_sq == 0:
        normalisation_factor = 1.0 # Or some fallback, if signal is completely zero there's no power
    else:
        normalisation_factor = 2 / (y.size * mean_y_sq)
    my_psd = [x*normalisation_factor for x in my_psd_]

    #make linear plot.
    ols_result = linear_fit(my_freq,my_psd,alpha=alpha)
    fig_linear = plot_linear_fit(my_freq,my_psd,alpha,ols_result,removed_psd,removed_freq,**kwargs)
    linear_slopes = { 'slope' : ols_result['slope']['result'],
     'confidence_interval' : ols_result['slope']['confidence_interval']}

    #make robust linear plot
    robust_ols_result = robust_linear_fit(my_freq, my_psd, alpha=alpha)
    fig_robust_linear = plot_linear_fit(my_freq,my_psd,alpha,robust_ols_result,removed_psd,removed_freq,legend_label = 'robust linear fit', title = 'Periodogram - robust linear fit')
    robust_linear_slopes = { 'slope' : robust_ols_result['slope']['result'],
     'confidence_interval' : robust_ols_result['slope']['confidence_interval']}

    #make segmented linear plot
    model_summaries,models = segmented_regression(my_freq,my_psd,max_breakpoints=max_breakpoints,n_boot=n_boot)
    fig_segmented = plot_segmented_fit(my_freq,my_psd,alpha,removed_psd,removed_freq,model_summaries,models,**kwargs)
    if len(model_summaries) > 0:
        segmented_slopes = {
            'breakpoint' : np.power(10, model_summaries[0]['estimates']['breakpoint1']['estimate']),
            'slope_less_than_breakpoint' : {
                                        'slope'               : model_summaries[0]['estimates']['alpha1']['estimate'],
                                        'confidence_interval' : model_summaries[0]['estimates']['alpha1']['confidence_interval'],
                                        },
            'slope_greater_than_breakpoint' : {
                                        'slope'               : model_summaries[0]['estimates']['alpha2']['estimate'],
                                        'confidence_interval' : model_summaries[0]['estimates']['alpha2']['confidence_interval'],
                                        }
            }

        #make robust segmented regression
        if segmented_slopes.get('breakpoint'):
            robust_segmented_results = robust_segmented_fit(my_freq,my_psd,segmented_slopes.get('breakpoint'),alpha)
            #plot robust segmented results
            if not bool(robust_segmented_results):
                fig_robust_segmented = None
            else:
                fig_robust_segmented = plot_robust_segmented_linear_fit(my_freq,my_psd, alpha,removed_psd,removed_freq,robust_segmented_results,**kwargs)
        else:
            segmented_slopes = {}
            robust_segmented_results = {}
            fig_robust_segmented = None

    else:
        segmented_slopes = {}
        robust_segmented_results = {}
        fig_robust_segmented = None


    #Hurst exponent
    all_hurst,detrended_hurst = calculate_hurst_exponent(x_trend,x_detrended)

    #rolling Hurst exponent
    #turning this off for now...
    rolling_hurst,rolling_hurst_detrended,fig_rolling_hurst = None,None,None
    # rolling_hurst,rolling_hurst_detrended = calculate_rolling_hurst_exponent(x_trend,x_detrended,window = hurst_window,**kwargs)
    # fig_rolling_hurst = plot_rolling_hurst(rolling_hurst,rolling_hurst_detrended,hurst_window)


    return fig_linear, fig_segmented, fig_robust_linear, linear_slopes, segmented_slopes, robust_linear_slopes,all_hurst,detrended_hurst, fig_rolling_hurst,rolling_hurst,rolling_hurst_detrended , fig_robust_segmented,robust_segmented_results


def generate_peridogram_result_only(
                            psd                    : np.array,
                            frequencies            : dict,
                            significant_components : list|None = None,
                            alpha                  : float = 0.05,
                            max_breakpoints        : int = 1,
                            n_boot                 : int = 500,
                            **kwargs):
    '''
    Function to run a periodogram analysis to find the fractal scaling of the time series.
    In all cases the trend is not considered, and significant periodic components can be ignored too using the input parameters.
    A scaled back version of generate_peridogram_plots.

    Parameters
    ----------
    psd : np.array
        DESCRIPTION. Power specrtal density from the CiSSA fit.
    frequencies : dict
        DESCRIPTION. Frequencies from the Cissa fit.
    significant_components : list|None, optional
        DESCRIPTION. The default is None. The default is None. A list of significant components which will not be considered in the periodogram analysis. Can also be None, in which case all components (except the trend) will be used for the periodogram, or if significant_components = None and monte_carlo_significant_components = True, then the monte carlo significant components will be removed.
    alpha : float, optional
        DESCRIPTION. The default is 0.05. Significance level for statistical tests.
    max_breakpoints : int, optional
        DESCRIPTION. The default is 1. Max number of breakpoints for the segmented linear fit. Currently will always be reset to 1 if >1.
    n_boot : int, optional
        DESCRIPTION. The default is 500. Number of bootstrap iterations for the segmented linear fit.
    **kwargs : TYPE
        DESCRIPTION. keyword arguments for segmented fitting.

    Returns
    -------
    linear_slopes : dict
        DESCRIPTION. Dictionary of results from linear fitting.
    robust_linear_slopes : dict
        DESCRIPTION. Dictionary of results from robust linear fitting.
    '''
    #get psd and frequencies of interest
    my_freq,my_psd,removed_psd,removed_freq = make_periodogram_arrays(psd, frequencies,significant_components=significant_components)

    #make robust linear plot
    robust_ols_result = robust_linear_fit(my_freq, my_psd, alpha=alpha)
    robust_linear_slopes = { 'slope' : robust_ols_result['slope']['result'],
     'confidence_interval' : robust_ols_result['slope']['confidence_interval']}

    #make segmented linear plot
    model_summaries,models = segmented_regression(my_freq,my_psd,max_breakpoints=max_breakpoints,n_boot=n_boot)
    if len(model_summaries) > 0:
        segmented_slopes = {
            'breakpoint' : np.power(10, model_summaries[0]['estimates']['breakpoint1']['estimate']),
            'slope_less_than_breakpoint' : {
                                        'slope'               : model_summaries[0]['estimates']['alpha1']['estimate'],
                                        'confidence_interval' : model_summaries[0]['estimates']['alpha1']['confidence_interval'],
                                        },
            'slope_greater_than_breakpoint' : {
                                        'slope'               : model_summaries[0]['estimates']['alpha2']['estimate'],
                                        'confidence_interval' : model_summaries[0]['estimates']['alpha2']['confidence_interval'],
                                        }
            }

        #make robust segmented regression
        if segmented_slopes.get('breakpoint'):
            robust_segmented_results = robust_segmented_fit(my_freq,my_psd,segmented_slopes.get('breakpoint'),alpha)
        else:
            segmented_slopes = {}
            robust_segmented_results = {}


    else:
        segmented_slopes = {}
        robust_segmented_results = {}


    return robust_linear_slopes,robust_segmented_results



###############################################################################
###############################################################################
def generate_lomb_scargle_peridogram_plots(
                            x_detrended            : np.ndarray,
                            psd                    : np.array,
                            frequencies            : dict,
                            significant_components : list|None = None,
                            alpha                  : float = 0.05,
                            max_breakpoints        : int = 1,
                            n_boot                 : int = 500,
                            normalization          : str = 'standard',
                            center_data            : bool = True,
                            fit_mean               : bool = True,
                            nterms                 : int = 1,
                            **kwargs):

    my_freq,my_psd,removed_psd,removed_freq = make_periodogram_arrays(psd, frequencies,significant_components=significant_components)


    ls_power = lomb_scargle_astropy(x_detrended,sorted(my_freq+removed_freq),
                              normalization=normalization,
                              center_data=center_data,
                              fit_mean=fit_mean,
                              nterms=nterms)


    ls_power_ = [x for x,y in zip(ls_power,sorted(my_freq+removed_freq)) if y not in removed_freq]

    #make robust linear plot
    robust_ols_result = robust_linear_fit(my_freq, ls_power_, alpha=alpha)
    fig_robust_linear = plot_linear_fit(my_freq,ls_power_,alpha,robust_ols_result,removed_psd,removed_freq,legend_label = 'robust linear fit', title = 'Periodogram - robust linear fit')
    robust_linear_slopes = { 'slope' : robust_ols_result['slope']['result'],
     'confidence_interval' : robust_ols_result['slope']['confidence_interval']}

    #make segmented linear plot
    model_summaries,models = segmented_regression(my_freq,ls_power_,max_breakpoints=max_breakpoints,n_boot=n_boot)
    fig_segmented = plot_segmented_fit(my_freq,ls_power_,alpha,removed_psd,removed_freq,model_summaries,models,**kwargs)
    if len(model_summaries) > 0:
        segmented_slopes = {
            'breakpoint' : np.power(10, model_summaries[0]['estimates']['breakpoint1']['estimate']),
            'slope_less_than_breakpoint' : {
                                        'slope'               : model_summaries[0]['estimates']['alpha1']['estimate'],
                                        'confidence_interval' : model_summaries[0]['estimates']['alpha1']['confidence_interval'],
                                        },
            'slope_greater_than_breakpoint' : {
                                        'slope'               : model_summaries[0]['estimates']['alpha2']['estimate'],
                                        'confidence_interval' : model_summaries[0]['estimates']['alpha2']['confidence_interval'],
                                        }
            }

        #make robust segmented regression
        if segmented_slopes.get('breakpoint'):
            robust_segmented_results = robust_segmented_fit(my_freq,ls_power_,segmented_slopes.get('breakpoint'),alpha)
            #plot robust segmented results
            fig_robust_segmented = plot_robust_segmented_linear_fit(my_freq,ls_power_, alpha,removed_psd,removed_freq,robust_segmented_results,**kwargs)
        else:
            segmented_slopes = {}
            robust_segmented_results = {}
            fig_robust_segmented = None

    else:
        segmented_slopes = {}
        robust_segmented_results = {}
        fig_robust_segmented = None




    return fig_segmented, fig_robust_linear, segmented_slopes, robust_linear_slopes, fig_robust_segmented,robust_segmented_results,ls_power




###############################################################################
###############################################################################




###############################################################################
###############################################################################
