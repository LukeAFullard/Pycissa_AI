import numpy as np
import copy
import matplotlib.pyplot as plt
import warnings
min_width = 720
min_height = 570
def get_surrogate_data(data:            np.ndarray,
                       L:               int,
                       psd:             np.ndarray,
                       Z:               np.ndarray,
                       results:         dict,
                       alpha:           float,
                       surrogates:      str,
                       sided_test:      str,
                       remove_trend:    bool,
                       frequencies:     dict,
                       alpha_slope:     float|None,
                       f_breakpoint:    float|None,
                       alpha_1_slope:   float|None,
                       alpha_2_slope:   float|None,
                       A_small_shuffle: float,
                       seed:            int|None) -> np.ndarray:
    '''
    Function which returns surrogate data for a provided time series.
    Currently supports the following types of surrogates:
        random_permutation: randomly shuffle the input data
        small_shuffle: the small shuffle method of Nakamura, T., & Small, M. (2005). Small-shuffle surrogate data: Testing for dynamics in fluctuating data with trends. Physical Review E, 72(5), 056216.
        ar1_fit: Fits an autoregressive model of order 1 to the data.

    Parameters
    ----------
    data : np.ndarray
        DESCRIPTION: Input data
    L : int
        DESCRIPTION: CiSSA window length.
    psd : np.ndarray
        DESCRIPTION: psd of each individual eigenvalue
    Z : np.ndarray
        DESCRIPTION: Output CiSSA results.
    alpha : float, optional
        DESCRIPTION: Significance level for surrogate hypothesis test. For example, --> 100*(1-alpha)% confidence interval. The default is 0.05 (a 95% confidence interval).
    results : dict
        DESCRIPTION: Dictionary of CiSSA results
    surrogates : str
        DESCRIPTION: Type of surrogates to fit. One of "random_permutation", "small_shuffle", "ar1_fit", "coloured_noise"
    sided_test : str, optional
        DESCRIPTION: When assessing the null hypothesis, are we running a one or two-sided test? The default is 'one sided'.
    remove_trend : bool, optional
        DESCRIPTION: Some surrogate methods make assumptions that are violated when there is a trend in the input data.
                        If remove_trend = True then the trend is removed before surrogates are generated, then added back to the surrogate data after generation. See  Lucio, J. H., Valdés, R., & Rodríguez, L. R. (2012). Improvements to surrogate data methods for nonstationary time series. Physical Review E, 85(5), 056202.
                        The default is True.
    frequencies : dict
        DESCRIPTION. Frequencies from the Cissa fit.
    alpha_slope : float|None,
        DESCRIPTION. Fitted slope of linear periodogram.
    f_breakpoint : float|None,
        DESCRIPTION. Breakpoint of segmented linear periodogram.
    alpha_1_slope : float|None,
        DESCRIPTION. Lower fitted slope of segmented linear periodogram.
    alpha_2_slope : float|None,
        DESCRIPTION. Upper fitted slope of segmented linear periodogram.
    A_small_shuffle : float
        DESCRIPTION: The parameter "A" in the small shuffle method. Not used for the other methods.
    seed : int|None
        DESCRIPTION: Random seed.

    Returns
    -------
    x_surrogate : np.ndarray
        DESCRIPTION: Surrogate for the input data.

    '''

    #check input 'surrogates' is a legitimate entry
    all_surrogate_types = ['random_permutation','small_shuffle','ar1_fit', 'coloured_noise_single', 'coloured_noise_segmented']
    if surrogates not in all_surrogate_types: raise ValueError(f"The parameter surrogates must be one of {all_surrogate_types}. You entered {surrogates}.")

    from pycissa.postprocessing.monte_carlo.surrogates import generate_random_permutation,generate_small_shuffle,generate_ar1_evenly
    if surrogates == 'random_permutation':
        x_surrogate = generate_random_permutation(data)
    if surrogates == 'small_shuffle':
        x_surrogate = generate_small_shuffle(data,A_small_shuffle)
    if surrogates == 'ar1_fit':
        x_surrogate = generate_ar1_evenly(data,seed)
    if surrogates in ['coloured_noise_single', 'coloured_noise_segmented']:
        from pycissa.postprocessing.monte_carlo.fractal_surrogates import generate_colour_surrogate
        x_surrogate = generate_colour_surrogate(data,alpha_slope,f_breakpoint,alpha_1_slope,alpha_2_slope,surrogates)


    return x_surrogate
###############################################################################
###############################################################################
def get_paired_psd(L:   int,
                   psd: np.ndarray
                   ) -> tuple[np.ndarray,int,int]:
    '''
    Function to calculate the matched psd.

    Parameters
    ----------
    L : int
        DESCRIPTION: CiSSA window length.
    psd : np.ndarray
        DESCRIPTION: psd of each individual eigenvalue

    Returns
    -------
    pzz : np.ndarray
        DESCRIPTION: Matched psd array (matching paired eigenvalues)
    nft : int
        DESCRIPTION: Size parameter.
    psd_length : int
        DESCRIPTION: Size of psd

    '''
    if np.mod(L,2):
        nf2 = (L+1)/2-1
    else:
        nf2 = L/2-1
    nft = int(nf2+np.abs(np.mod(L,2)-2)  )
    psd_length = len(psd)
    if np.mod(psd_length,2):
        pzz = np.append(psd[0], 2*psd[1:nft])
    else:
        pzz = np.append(
            np.append(psd[0], 2*psd[1:nft-1]),
            psd[nft-1]
            )
    return pzz,nft,psd_length
###############################################################################
###############################################################################
def calculate_surrogate_psd(x_surrogate:              np.ndarray,
                            L:                        int,
                            extension_type:           str,
                            multi_thread_run:         bool,
                            generate_toeplitz_matrix: bool,
                            psd_length:               int,
                            nft:                      int,):
    '''
    Function to approximate the psd of the surrogate data.
    Applies the CiSSA algorithm to the surrogate data but does not perform grouping (the sloest part of the CiSSA algorithm).
    This leads to a significant speed up compared to applying the full CiSSA.

    Parameters
    ----------
    x_surrogate : np.ndarray
        DESCRIPTION: Input surrogate data
    L : int
        DESCRIPTION: CiSSA window length.
    extension_type : str
        DESCRIPTION: extension type for left and right ends of the time series. The default is AR_LR
    multi_thread_run : bool
        DESCRIPTION: Flag to indicate whether the diagonal averaging is performed on multiple cpu cores (True) or not.
    generate_toeplitz_matrix : bool
        DESCRIPTION: Flag to indicate whether we need to calculate the symmetric Toeplitz matrix or not.
    psd_length : int
        DESCRIPTION: Length of the psd (calculated in the get_paired_psd function)
    nft : int
        DESCRIPTION: Size parameter (calculated in the get_paired_psd function)

    Returns
    -------
    pzz_surrogate : np.ndarray
        DESCRIPTION: surrogate psd.

    '''
    from pycissa.processing.matrix_operations.matrix_operations import run_cissa_psd_step
    #calculate psd
    psd_surrogate = run_cissa_psd_step(x_surrogate,
                  L=L,
                  extension_type=extension_type,
                  multi_thread_run=multi_thread_run,
                  generate_toeplitz_matrix=generate_toeplitz_matrix
                  )
    if np.mod(psd_length,2):
        pzz_surrogate = np.append(psd_surrogate[0], 2*psd_surrogate[1:nft])
    else:
        pzz_surrogate = np.append(
            np.append(psd_surrogate[0], 2*psd_surrogate[1:nft-1]),
            psd_surrogate[nft-1]
            )
    return pzz_surrogate

###############################################################################
###############################################################################
def plot_monte_carlo_results(plot_period:                      list,
                             surrogate_psd:                    list,
                             signal_psd:                       list,
                             alpha:                            float,
                             significant_components_index:     list,
                             non_significant_components_index: list,
                             log_scale: bool = True) -> tuple[plt.figure,plt.axes]:
    '''
    Function to plot the surrogate psd vs the signal psd

    Parameters
    ----------
    plot_period : list
        DESCRIPTION: List of non-dimensional periods = 1/frequency
    surrogate_psd : list
        DESCRIPTION: list of surrogate psd values
    signal_psd : list
        DESCRIPTION: psd of the signal
    alpha : float
        DESCRIPTION: significance level
    significant_components_index : list
        DESCRIPTION: Components identified to be significant
    non_significant_components_index : list
        DESCRIPTION: Components identified to not be significant

    Returns
    -------
    fig : matplotlib figure
        DESCRIPTION: Figure
    ax : matplotlib axes
        DESCRIPTION: Axes

    '''
    #create a plot of the results including significance
    fig, ax = plt.subplots()
    #surrogate results
    ax.plot(plot_period, surrogate_psd, linestyle='--',color='red', alpha=0.5, label=f'Surrogate upper tolerance interval, alpha = {alpha}')
    ax.scatter(np.array(plot_period)[significant_components_index], np.array(signal_psd)[significant_components_index], color='blue', label='Significant components')
    ax.scatter(np.array(plot_period)[non_significant_components_index], np.array(signal_psd)[non_significant_components_index], color='black', label='Non-significant components')
    if log_scale:
        ax.set_yscale('log')
        ax.set_xscale('log')

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
    return fig,ax
###############################################################################
###############################################################################
def check_for_significance(result:                   dict,
                           K_surrogates:             int,
                           alpha:                    float,
                           pzz:                      np.ndarray,
                           surrogate_results:        list,
                           surrogates:               str,
                           trend_always_significant: bool)-> tuple[dict,list,list,list]:
    '''
    Function to check signal components for significance vs the surrogate data.

    Parameters
    ----------
    result : dict
        DESCRIPTION: Dictionary of results
    K_surrogates : int
        DESCRIPTION: Multiplier for number of surrogates. Number of surrogate data allowed to be larger than the signal and signal to still be significant = K_surrogates - 1
    alpha : float
        DESCRIPTION: significance level
    pzz : np.ndarray
        DESCRIPTION: psd of signal
    surrogate_results : list
        DESCRIPTION: list of surrogate results
    surrogates : str
        DESCRIPTION: Type of surrogates
    trend_always_significant : bool
        DESCRIPTION: Is the trend always significant? Needed as some surrogate methods test for short fluctuations, not long ones such as trend.

    Returns
    -------
    result_ : dict
        DESCRIPTION: Results dictionary
    plot_period : list
        DESCRIPTION: List of non-dimensional periods = 1/frequency
    surrogate_psd : list
        DESCRIPTION: psd of surrogate data
    signal_psd : list
        DESCRIPTION: psd of signal

    '''
    plot_period = []
    signal_psd       = []
    surrogate_psd    = []
    result_ = copy.deepcopy(result)
    for results_key_k in result_.get('components').keys():
        allowed_larger_surrogates = K_surrogates - 1
        key_array_position = result_.get('components').get(results_key_k).get('array_position')
        psd_signal = pzz[key_array_position]

        #find how many of the surogate data series have a larger psd than the original signal
        larger_surrogates = [x for x in surrogate_results.get(results_key_k) if x > psd_signal]

        #update results dictionary and return it
        result_.get('components').get(results_key_k).setdefault('monte_carlo', {})
        result_.get('components').get(results_key_k).get('monte_carlo').setdefault(surrogates, {})
        result_.get('components').get(results_key_k).get('monte_carlo').get(surrogates).setdefault('alpha', {})

        result_.get('components').get(results_key_k).get('monte_carlo').get(surrogates).get('alpha').setdefault(alpha,{})

        if len(larger_surrogates) > allowed_larger_surrogates:
            result_.get('components').get(results_key_k).get('monte_carlo').get(surrogates).get('alpha').get(alpha).update({'signal_psd':psd_signal,
                                                                                                    'surrogate_psd':surrogate_results.get(results_key_k),
                                                                                                    'pass'         :False})
        else:
            result_.get('components').get(results_key_k).get('monte_carlo').get(surrogates).get('alpha').get(alpha).update({'signal_psd':psd_signal,
                                                                                                    'surrogate_psd':surrogate_results.get(results_key_k),
                                                                                                'pass'         :True})

        #build list for plotting
        sorted_surrogates = sorted(surrogate_results.get(results_key_k))
        surrogate_index = -1 - allowed_larger_surrogates
        plot_period.append(result_.get('components').get(results_key_k).get('unitless period (number of timesteps)'))
        signal_psd.append(psd_signal)
        surrogate_psd.append(sorted_surrogates[surrogate_index])
    if trend_always_significant:
        result_.get('components').get('trend').get('monte_carlo').get(surrogates).get('alpha').get(alpha).update({'pass':True})

    return result_,plot_period,surrogate_psd,signal_psd

###############################################################################
###############################################################################


###############################################################################
###############################################################################
def find_significant_components(result:     dict,
                                surrogates: str,
                                alpha:      float) -> tuple[list,list]:
    '''
    Function to find index of significant components

    Parameters
    ----------
    result : dict
        DESCRIPTION: Results dictionary
    surrogates : str
        DESCRIPTION: Type of surrogate used
    alpha : float
        DESCRIPTION: Significance level

    Returns
    -------
    significant_components_index : list
        DESCRIPTION: List of index of significant components
    non_significant_components_index : list
        DESCRIPTION: List of index of non-significant components

    '''
    significant_components_index = []
    non_significant_components_index = []
    for iter_i,entry_i in enumerate(result.get('components').keys()):
        if result.get('components').get(entry_i).get('monte_carlo').get(surrogates).get('alpha').get(alpha).get('pass'):
            #significant components
            significant_components_index.append(iter_i)
        else:
            #non-significant components
            non_significant_components_index.append(iter_i)

    return significant_components_index,    non_significant_components_index
###############################################################################
###############################################################################
def run_monte_carlo_test(x:                        np.ndarray,
                         L:                        int,
                         psd:                      np.ndarray,
                         Z:                        np.ndarray,
                         results:                  dict,
                         frequencies:              dict,
                         alpha:                    float = 0.05,
                         K_surrogates:             int = 1,
                         surrogates:               str = 'random_permutation',
                         seed:                     int|None = None,
                         sided_test:               str = 'one sided',
                         remove_trend:             bool = True,
                         trend_always_significant: bool = True,
                         A_small_shuffle:          float = 1.,
                         extension_type:           str = 'AR_LR',
                         multi_thread_run:         bool = False,
                         generate_toeplitz_matrix: bool = False,
                         plot_figure:              bool = True,
                         ) -> tuple[dict,plt.figure]:
    '''
    Function to run a monte_carlo significance test on components of a signal, extracted via CiSSA.
    Signal psd/eigenvalues are compared to those obtained by applying CiSSA to surrogate data.
    Surrogates are generated using one of three available algorithms:
        random_permutation: randomly shuffle the input data
        small_shuffle: the small shuffle method of Nakamura, T., & Small, M. (2005). Small-shuffle surrogate data: Testing for dynamics in fluctuating data with trends. Physical Review E, 72(5), 056216.
        ar1_fit: Fits an autoregressive model of order 1 to the data.

    Parameters
    ----------
    x : np.ndarray
        DESCRIPTION: Input array
    L : int
        DESCRIPTION: CiSSA window length.
    psd : np.ndarray
        DESCRIPTION: psd of each individual eigenvalue
    Z : np.ndarray
        DESCRIPTION: Output CiSSA results.
    results : dict
        DESCRIPTION: Dictionary of CiSSA results
    frequencies : dict
        DESCRIPTION. Frequencies from the Cissa fit.
    alpha : float, optional
        DESCRIPTION: Significance level for surrogate hypothesis test. For example, --> 100*(1-alpha)% confidence interval. The default is 0.05 (a 95% confidence interval).
    K_surrogates : int, optional
        DESCRIPTION: Multiplier for number of surrogates. Number of surrogate data allowed to be larger than the signal and signal to still be significant = K_surrogates - 1.
                        For a one-sided test, the number of surrogate data series generated is K_surrogates/alpha - 1. For a two sided test it is 2*K_surrogates/alpha - 1.
                        The default is 1.
    surrogates : str, optional
        DESCRIPTION: The type of surrogates to generate for the hypothesis test.
                        One of "random_permutation", "small_shuffle", "ar1_fit", "coloured_noise_single", "coloured_noise_segmented".
                        The default is 'random_permutation'.
    seed : int|None, optional
        DESCRIPTION: Random seed for reproducability. The default is None.
    sided_test : str, optional
        DESCRIPTION: When assessing the null hypothesis, are we running a one or two-sided test? The default is 'one sided'.
    remove_trend : bool, optional
        DESCRIPTION: Some surrogate methods make assumptions that are violated when there is a trend in the input data.
                        If remove_trend = True then the trend is removed before surrogates are generated, then added back to the surrogate data after generation. See  Lucio, J. H., Valdés, R., & Rodríguez, L. R. (2012). Improvements to surrogate data methods for nonstationary time series. Physical Review E, 85(5), 056202.
                        The default is True.
    trend_always_significant : bool, optional
        DESCRIPTION: Option to ensure the trend is always significant. (Possibly) necessary if remove_trend = True.The default is True.
    A_small_shuffle : float, optional
        DESCRIPTION: If surrogates = 'small_shuffle', then this parameter is the "A" parameter in the small shuffle paper, Nakamura, T., & Small, M. (2005). Small-shuffle surrogate data: Testing for dynamics in fluctuating data with trends. Physical Review E, 72(5), 056216.
                        The default is 1.
    extension_type : str, optional
        DESCRIPTION: extension type for left and right ends of the time series. The default is AR_LR.
    multi_thread_run : bool, optional
        DESCRIPTION: Flag to indicate whether the diagonal averaging is performed on multiple cpu cores (True) or not. The default is False.
    generate_toeplitz_matrix : bool, optional
        DESCRIPTION: Flag to indicate whether we need to calculate the symmetric Toeplitz matrix or not. The default is False.
    plot_figure: bool, optional
        DESCRIPTION: Flag to indicate whether we need to plot results or not. The default is True.
    Returns
    -------
    result : dict
        DESCRIPTION: Results dictionary.
    fig : plt.figure
        DESCRIPTION: Figure of surrogate psd vs signal to show significant components.
    '''

    #check input 'sided_test' and calculate number of surrogates required
    if sided_test == 'one sided':
        number_of_surrogates = int(K_surrogates/alpha - 1)
    elif sided_test == 'two sided':
        number_of_surrogates = int(2*K_surrogates/alpha - 1)
    else:raise ValueError(f"The parameter sided_test must be one of 'one sided' or 'two sided'. You entered '{sided_test}'")

    ############################################
    result = copy.deepcopy(results)
    x_copy = copy.deepcopy(x)
    if remove_trend:
        x_copy -= result.get('components').get('trend').get('reconstructed_data').reshape(x_copy.shape)

    #get psd
    pzz,nft,psd_length = get_paired_psd(L,psd)

    alpha_slope,f_breakpoint,alpha_1_slope,alpha_2_slope = None,None,None,None
    if surrogates in ['coloured_noise_single', 'coloured_noise_segmented']:
        from pycissa.postprocessing.monte_carlo.fractal_surrogates import prepare_for_coloured_surrogates
        alpha_slope,f_breakpoint,alpha_1_slope,alpha_2_slope = prepare_for_coloured_surrogates(x_copy,L,psd,Z,results,alpha,surrogates,sided_test,remove_trend,frequencies,seed)



    #iterate through the surrogates
    surrogate_results = {}
    for surrogate_i in range(0,number_of_surrogates):
        #generate surrogates
        x_surrogate = get_surrogate_data(x_copy,L,psd,Z,result,alpha,surrogates,sided_test,remove_trend,frequencies,alpha_slope,f_breakpoint,alpha_1_slope,alpha_2_slope,A_small_shuffle,seed)


        #add the trend back in
        if remove_trend:
            # Lucio, J. H., Valdés, R., & Rodríguez, L. R. (2012). Improvements to surrogate data methods for nonstationary time series. Physical Review E, 85(5), 056202.
            # Unfortunately this seems to make the trend not significant.
            # Can offset this by setting trend_always_significant = True
            x_surrogate += result.get('components').get('trend').get('reconstructed_data').reshape(x_surrogate.shape)


        #calculate surrogate psd
        pzz_surrogate = calculate_surrogate_psd(x_surrogate,L,extension_type,multi_thread_run,generate_toeplitz_matrix,psd_length,nft)

        #add psd into surrogate_results dictionary
        for results_key_j in result.get('components').keys():
            key_array_position = result.get('components').get(results_key_j).get('array_position')
            surrogate_results.setdefault(results_key_j, []).append(pzz_surrogate[key_array_position])
    ############################################
    #check each psd for significance
    result,plot_period,surrogate_psd,signal_psd = check_for_significance(result,K_surrogates,alpha,pzz,surrogate_results,surrogates,trend_always_significant)

    #find the significant components
    significant_components_index,    non_significant_components_index = find_significant_components(result,surrogates,alpha)
    fig = None
    if plot_figure:
        #create a plot of the results including significance
        fig,ax = plot_monte_carlo_results(plot_period,surrogate_psd,signal_psd,alpha,significant_components_index,non_significant_components_index)

    return result,fig


###############################################################################
###############################################################################
def prepare_monte_carlo_kwargs(kw_dict):
    K_surrogates = kw_dict.get('K_surrogates')
    if not K_surrogates: K_surrogates = 1

    surrogates = kw_dict.get('surrogates')
    if not surrogates: surrogates = 'random_permutation'

    seed = kw_dict.get('seed')

    sided_test = kw_dict.get('sided_test')
    if not sided_test: sided_test = 'one sided'

    remove_trend = kw_dict.get('remove_trend')
    if remove_trend == None: remove_trend = True

    trend_always_significant = kw_dict.get('trend_always_significant')
    if trend_always_significant == None: trend_always_significant = True

    A_small_shuffle = kw_dict.get('A_small_shuffle')
    if not A_small_shuffle: A_small_shuffle = 1.

    # extension_type = kw_dict.get('extension_type')
    # if not extension_type: extension_type = 'AR_LR'

    # multi_thread_run = kw_dict.get('multi_thread_run')
    # if not multi_thread_run: multi_thread_run = True

    generate_toeplitz_matrix = kw_dict.get('generate_toeplitz_matrix')
    if generate_toeplitz_matrix == None: generate_toeplitz_matrix = False

    return K_surrogates, surrogates, seed, sided_test, remove_trend,trend_always_significant, A_small_shuffle, generate_toeplitz_matrix

###############################################################################
###############################################################################



###############################################################################
###############################################################################



###############################################################################
###############################################################################