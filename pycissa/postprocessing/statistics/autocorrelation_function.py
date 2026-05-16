import numpy as np
min_width = 720
min_height = 570
###############################################################################
###############################################################################
###############################################################################
def calculate_autocorrelation_function(x:                np.ndarray,
                                       adjusted:         bool = False,
                                       nlags:            int|None = None,
                                       qstat:            bool = False,
                                       fft:              bool = True,
                                       alpha:            float = 0.05,
                                       bartlett_confint: bool = True,
                                       missing:          str = 'none'):
    '''
    Wrapper of statsmodels acf to calculate autocorrelation function.
    [1]
    Parzen, E., 1963. On spectral analysis with missing observations and amplitude modulation. Sankhya: The Indian Journal of Statistics, Series A, pp.383-392.

    [2]
    Brockwell and Davis, 1987. Time Series Theory and Methods

    [3]
    Brockwell and Davis, 2010. Introduction to Time Series and Forecasting, 2nd edition.

    Parameters
    ----------
    x : np.ndarray
        DESCRIPTION: The time series data.
    adjusted : bool, optional
        DESCRIPTION: If True, then denominators for autocovariance are n-k, otherwise n. The default is False.
    nlags : int|None, optional
        DESCRIPTION: Number of lags to return autocorrelation for. If not provided, uses min(10 * np.log10(nobs), nobs - 1). The returned value includes lag 0 (ie., 1) so size of the acf vector is (nlags + 1,). The default is None.
    qstat : bool, optional
        DESCRIPTION: If True, returns the Ljung-Box q statistic for each autocorrelation coefficient. See q_stat for more information. The default is False.
    fft : bool, optional
        DESCRIPTION: If True, computes the ACF via FFT. The default is True.
    alpha : float, optional
        DESCRIPTION: If a number is given, the confidence intervals for the given level are returned. For instance if alpha=.05, 95 % confidence intervals are returned where the standard deviation is computed according to Bartlett”s formula. The default is 0.05.
    bartlett_confint : bool, optional
        DESCRIPTION: Confidence intervals for ACF values are generally placed at 2 standard errors around r_k. The formula used for standard error depends upon the situation. If the autocorrelations are being used to test for randomness of residuals as part of the ARIMA routine, the standard errors are determined assuming the residuals are white noise. The approximate formula for any lag is that standard error of each r_k = 1/sqrt(N). See section 9.4 of [2] for more details on the 1/sqrt(N) result. For more elementary discussion, see section 5.3.2 in [3]. For the ACF of raw data, the standard error at a lag k is found as if the right model was an MA(k-1). This allows the possible interpretation that if all autocorrelations past a certain lag are within the limits, the model might be an MA of order defined by the last significant autocorrelation. In this case, a moving average model is assumed for the data and the standard errors for the confidence intervals should be generated using Bartlett’s formula. For more details on Bartlett formula result, see section 7.2 in [2]. The default is True.
    missing : str, optional
        DESCRIPTION: A string in [“none”, “raise”, “conservative”, “drop”] specifying how the NaNs are to be treated. “none” performs no checks. “raise” raises an exception if NaN values are found. “drop” removes the missing observations and then estimates the autocovariances treating the non-missing as contiguous. “conservative” computes the autocovariance using nan-ops so that nans are removed when computing the mean and cross-products that are used to estimate the autocovariance. When using “conservative”, n is set to the number of non-missing observations. The default is 'none'.

    Returns
    -------
    acf : np.ndarray
        DESCRIPTION: The autocorrelation function for lags 0, 1, …, nlags. Shape (nlags+1,).
    confidence_interval : np.ndarray
        DESCRIPTION: Confidence intervals for the ACF at lags 0, 1, …, nlags. Shape (nlags + 1, 2). Returned if alpha is not None.
    q_stat : np.ndarray
        DESCRIPTION: The Ljung-Box Q-Statistic for lags 1, 2, …, nlags (excludes lag zero). Returned if q_stat is True.
    p_values : np.ndarray
        DESCRIPTION: The p-values associated with the Q-statistics for lags 1, 2, …, nlags (excludes lag zero). Returned if q_stat is True.

    '''
    from statsmodels.tsa.stattools import acf
    acf,confidence_interval,q_stat,p_values = acf(x=x,
                                                  adjusted=adjusted,
                                                  nlags=nlags,
                                                  qstat=qstat,
                                                  fft=fft,
                                                  alpha=alpha,
                                                  bartlett_confint=bartlett_confint,
                                                  missing=missing)
    return acf,confidence_interval,q_stat,p_values

###############################################################################
###############################################################################
###############################################################################
def calculate_partial_autocorrelation_function(x:      np.ndarray,
                                               nlags:  int|None = None,
                                               method: str = 'ywadjusted',
                                               alpha:  float = 0.05):
    '''
    Wrapper of statsmodels acf to calculate partial autocorrelation function.

    Parameters
    ----------
    x : np.ndarray
        DESCRIPTION: Observations of time series for which pacf is calculated.
    nlags : int|None, optional
        DESCRIPTION: Number of lags to return autocorrelation for. If not provided, uses min(10 * np.log10(nobs), nobs // 2 - 1). The returned value includes lag 0 (ie., 1) so size of the pacf vector is (nlags + 1,). The default is None.
    method : str, optional
        DESCRIPTION: Specifies which method for the calculations to use.
                    “yw” or “ywadjusted” : Yule-Walker with sample-size adjustment in denominator for acovf. Default.
                    “ywm” or “ywmle” : Yule-Walker without adjustment.
                    “ols” : regression of time series on lags of it and on constant.
                    “ols-inefficient” : regression of time series on lags using a single common sample to estimate all pacf coefficients.
                    “ols-adjusted” : regression of time series on lags with a bias adjustment.
                    “ld” or “ldadjusted” : Levinson-Durbin recursion with bias correction.
                    “ldb” or “ldbiased” : Levinson-Durbin recursion without bias correction.
                    “burg” : Burg”s partial autocorrelation estimator.
                    The default is 'ywadjusted'.
    alpha : float, optional
        DESCRIPTION. If a number is given, the confidence intervals for the given level are returned. For instance if alpha=.05, 95 % confidence intervals are returned where the standard deviation is computed according to 1/sqrt(len(x)). The default is 0.05.

    Returns
    -------
    pact : np.ndarray
        DESCRIPTION: The partial autocorrelations for lags 0, 1, …, nlags. Shape (nlags+1,).
    confidence_interval : np.ndarray
        DESCRIPTION: Confidence intervals for the PACF at lags 0, 1, …, nlags. Shape (nlags + 1, 2). Returned if alpha is not None.

    '''
    from statsmodels.tsa.stattools import pacf
    pact,confidence_interval = pacf(x=x,
                                    nlags=nlags,
                                    method=method,
                                    alpha=alpha)
    return pact,confidence_interval



###############################################################################
###############################################################################
###############################################################################
import matplotlib.pyplot as plt
def plot_time_series_and_acf_pacf(t:                np.ndarray,
                                  x:                np.ndarray,
                                  acf_lags:         int|list|None=None,
                                  pacf_lags:        int|list|None=None,
                                  alpha:            float=0.05,
                                  use_vlines:       bool=True,
                                  adjusted:         bool=False,
                                  fft:              bool=False,
                                  missing:          str='none',
                                  zero:             bool=True,
                                  auto_ylims:       bool=False,
                                  bartlett_confint: bool=True,
                                  pacf_method:      str='ywm',
                                  acf_color:        str='blue',
                                  pacf_color:       str='blue',
                                  title:            str|None=None,
                                  title_size:       float|int=14,
                                  label_size:       float|int=12
                                  ):
    '''
    Function to generate and return a figure with three subplots:
        1) time series data,
        2) the autocorrelation function of the data,
        3) the partial autocorrelation function of the data.

    Parameters
    ----------
    t : np.ndarray
        DESCRIPTION: array of input times/dates.
    x : np.ndarray
        DESCRIPTION: array of input data.
    acf_lags : int|list|None, optional
        DESCRIPTION. The default is None. An int or array of acf lag values, used on horizontal axis. If None, half the time series length is used as lags.
    pacf_lags : int|list|None, optional
        DESCRIPTION. The default is None. An int or array of acf lag values, used on horizontal axis. If None, acf_lags is used as lags.
    alpha : float, optional
        DESCRIPTION. The default is 0.05. If a number is given, the confidence intervals for the given level are returned. For instance if alpha=.05, 95 % confidence intervals are returned where the standard deviation is computed according to Bartlett’s formula. If None, no confidence intervals are plotted.
    use_vlines : bool, optional
        DESCRIPTION. The default is True. If True, vertical lines and markers are plotted. If False, only markers are plotted.
    adjusted : bool, optional
        DESCRIPTION. The default is False. For acf, if True, then denominators for autocovariance are n-k, otherwise n
    fft : bool, optional
        DESCRIPTION. The default is False. For acf, if True, computes the ACF via FFT.
    missing : str, optional
        DESCRIPTION. The default is 'none'. A string in [‘none’, ‘raise’, ‘conservative’, ‘drop’] specifying how the NaNs are to be treated.
    zero : bool, optional
        DESCRIPTION. The default is True. Flag indicating whether to include the 0-lag autocorrelation.
    auto_ylims : bool, optional
        DESCRIPTION. The default is False. If True, adjusts automatically the y-axis limits to ACF values.
    bartlett_confint : bool, optional
        DESCRIPTION. The default is True. Confidence intervals for ACF values are generally placed at 2 standard errors around r_k. The formula used for standard error depends upon the situation. If the autocorrelations are being used to test for randomness of residuals as part of the ARIMA routine, the standard errors are determined assuming the residuals are white noise. The approximate formula for any lag is that standard error of each r_k = 1/sqrt(N). See section 9.4 of [1] for more details on the 1/sqrt(N) result. For more elementary discussion, see section 5.3.2 in [2]. For the ACF of raw data, the standard error at a lag k is found as if the right model was an MA(k-1). This allows the possible interpretation that if all autocorrelations past a certain lag are within the limits, the model might be an MA of order defined by the last significant autocorrelation. In this case, a moving average model is assumed for the data and the standard errors for the confidence intervals should be generated using Bartlett’s formula. For more details on Bartlett formula result, see section 7.2 in [1]. [1] Brockwell and Davis, 1987. Time Series Theory and Methods [2] Brockwell and Davis, 2010. Introduction to Time Series and Forecasting, 2nd edition.
    pacf_method : str, optional
        DESCRIPTION. The default is 'ywm'. Specifies which method for the calculations to use:
                                            “ywm” or “ywmle” : Yule-Walker without adjustment. Default.
                                            “yw” or “ywadjusted” : Yule-Walker with sample-size adjustment in denominator for acovf. Default.
                                            “ols” : regression of time series on lags of it and on constant.
                                            “ols-inefficient” : regression of time series on lags using a single common sample to estimate all pacf coefficients.
                                            “ols-adjusted” : regression of time series on lags with a bias adjustment.
                                            “ld” or “ldadjusted” : Levinson-Durbin recursion with bias correction.
                                            “ldb” or “ldbiased” : Levinson-Durbin recursion without bias correction.
    acf_color : str, optional
        DESCRIPTION. The default is 'blue'. Colour of the acf plot markers.
    pacf_color : str, optional
        DESCRIPTION. The default is 'blue'. Colour of the pacf plot markers.
    title : str|None, optional
        DESCRIPTION. The default is None. Title for the figure data figure
    title_size : float|int, optional
        DESCRIPTION. The default is 14.
    label_size : float|int, optional
        DESCRIPTION. The default is 12.

    Returns
    -------
    fig : matplotlib figure
        DESCRIPTION. Figure of time series, acf, pacf.

    '''

    import numpy as np
    import statsmodels.graphics.tsaplots as tsaplots

    #if the acf lags are not set then take as half the time series length
    if acf_lags is None:
        acf_lags = int(np.floor(len(x)/2))
    #if the pacf lags are not set, take as the same as acf
    if pacf_lags is None:
        pacf_lags = acf_lags

    # Create a figure and axes
    fig, ax = plt.subplots(3, 1, figsize=(10, 15), sharex=False)

    # Plot raw time series data
    ax[0].plot(t, x, label='Time Series Data', color='black')
    if title is None:
        ax[0].set_title('Raw Time Series Data', fontsize=title_size)
    else:
        ax[0].set_title(title, fontsize=title_size)
    ax[0].set_ylabel('Value', fontsize=label_size)
    ax[0].set_xlabel('Time', fontsize=label_size)

    # Plot ACF
    tsaplots.plot_acf(x,
                      lags=acf_lags,
                      ax=ax[1],
                      color=acf_color,
                      alpha=alpha,
                      use_vlines=use_vlines,
                      adjusted=adjusted,
                      fft=fft,
                      missing=missing,
                      zero=zero,
                      auto_ylims=auto_ylims,
                      bartlett_confint=bartlett_confint,
                      vlines_kwargs = {'colors':'black',
                                       'linewidths':(0.5,)}
                      )
    ax[1].set_title('Autocorrelation Function (ACF)', fontsize=title_size)


    # Plot PACF
    tsaplots.plot_pacf(x,
                       lags=pacf_lags,
                       ax=ax[2],
                       color=pacf_color,
                       alpha=alpha,
                       method = pacf_method,
                       use_vlines=use_vlines,
                       vlines_kwargs = {'colors':'black',
                                        'linewidths':(0.5,)},
                       zero=zero)
    ax[2].set_title('Partial Autocorrelation Function (PACF)', fontsize=title_size)


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
