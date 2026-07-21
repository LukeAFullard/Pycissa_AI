import numpy as np
import matplotlib.pyplot as plt
import warnings
min_width = 720
min_height = 570
###############################################################################
###############################################################################


###############################################################################
###############################################################################
def find_last_true(lst:list) -> int:
    '''
    function to find the index of the last True entry in a list

    Parameters
    ----------
    lst : list
        DESCRIPTION: Input list

    Returns
    -------
    int
        DESCRIPTION: index of the last True entry in a list, otherwise -1

    '''

    for i in range(len(lst)-1, -1, -1):
        if lst[i] == True:
            return i
    return -1

###############################################################################
###############################################################################
def validate_input_parameters(x:              np.ndarray,
                              L:              int,
                              extension_type: str,
                              outliers:       list,
                              initial_guess:  list,
                              convergence:    list
                              ) -> np.ndarray:
    '''
    Function to check input parameters for the gap filling method

    Parameters
    ----------
    x : np.ndarray
        DESCRIPTION: Input array
    L : int
        DESCRIPTION: CiSSA window length.
    extension_type : str
        DESCRIPTION: extension type for left and right ends of the time series
    outliers : list
        DESCRIPTION: How to find outliers/missing values and the threshold value.
                     Current options are:
                         0) ['nan_only',None]  -- classified all NaN values as outliers/missing data
                         1) ['<',threshold]  -- classifies all values below the threshold as outliers
                         2) ['>',threshold]  -- classifies all values above the threshold as outliers
                         3) ['<>',[low_threshold, hi_threshold]]  -- classifies all values not between the two thresholds as outliers
                         4) ['k',multiplier]  -- classifies all values above the multiplier of the median average deviation as outliers. IMPORTANT NOTE: Does not converge very well/at all if there are consecutive missing values.
    initial_guess : list
        DESCRIPTION: How to choose the initial guess for the missing data/outliers.
                     Current options are:
                         1) ['max', ''] -- Initial guess is the maximum of the time series (ignoring outliers)
                         2) ['median', ''] -- Initial guess is the median of the time series
                         3) ['value', numeric] -- Initial guess is the provided numeric value, the second entry in the list.
                         4) ['previous', numeric] -- Initial guess is the previous value of the time series (ignoring any outliers) multiplied by the numeric value, the second entry in the input list.
    convergence : list
        DESCRIPTION: How to define the convergence of the outlier fitting method.
                     Current options are:
                         1) ['value', threshold] -- convergence error must be less than the threshold value to signify convergence
                         2) ['min', multiplier]  -- convergence error must be less than the multiplier*(minimum non-outlier value of the data) to signify convergence
                         3) ['med', multiplier]  -- convergence error must be less than the multiplier*(median non-outlier value of the data) to signify convergence

    Returns
    -------
    x : np.ndarray
        DESCRIPTION: Possibky reshapred inout data

    '''
    #check L is an integer, extension_type a string
    if not type(extension_type) == str:
        raise TypeError('Input parameter "extension_type" should be a string')
    if not type(L) == int:
        raise TypeError('Input parameter "L" should be an integer')

    #check x is a numpy array
    if not type(x) is np.ndarray:
        try:
            x = np.array(x)
            x = x.reshape(len(x),1)
        except: raise ValueError(f'Input "x" is not a numpy array, nor can be converted to one.')
    myshape = x.shape
    if len(myshape) == 2:
        rows, cols = myshape[0],myshape[1]
    else:
        try:
            x = x.reshape(len(x),1)
            rows, cols = x.shape
        except:
            raise ValueError(f'Input "x" should be a column vector (i.e. only contain a single column). The size of x is ({myshape})')

    #check outliers, errors are length 2 lists
    if not type(outliers) is list:
        raise TypeError('Input parameter "outliers" should be a list')
    if not len(outliers) == 2:
        raise ValueError('Input parameter "outliers" should be a length 2 list')
    if not type(initial_guess) is list:
        raise TypeError('Input parameter "initial_guess" should be a list')
    if not len(initial_guess) == 2:
        raise ValueError('Input parameter "initial_guess" should be a length 2 list')
    if not type(convergence) is list:
        raise TypeError('Input parameter "convergence" should be a list')
    if not len(convergence) == 2:
        raise ValueError('Input parameter "convergence" should be a length 2 list')

    return x

###############################################################################
###############################################################################
def initialise_error_estimates(estimate_error: bool):
    '''
    Function to initialise some of the required parameters for error estimation.

    Parameters
    ----------
    estimate_error : bool
        DESCRIPTION: Flag which determines if we will be estimating the error in the gap filling or not

    Returns
    -------
    error_rmse : float
        DESCRIPTION: Root mean squared error of test points.
    error_rmse_percentage : float
        DESCRIPTION: Percentage root mean squared error of test points.
    fig : matplotlib.figure
        DESCRIPTION: Figure object for plotting a figure showing imputed data points.
    ax : matplotlib.axes
        DESCRIPTION: Axes object for plotting a figure showing imputed data points.
    error_estimates : np.ndarray
        DESCRIPTION: Array of errors associated with the test points
    error_estimates_percentage : np.ndarray
        DESCRIPTION: Array of percentage errors associated with the test points.
    original_points : np.ndarray
        DESCRIPTION: Array of the original time series points
    imputed_points : np.ndarray
        DESCRIPTION: Array of the imputed time series points.

    '''
    error_rmse                 = np.nan
    error_rmse_percentage      = np.nan
    fig, ax                    = plt.subplots(1, 1)

    if not estimate_error:
        #if we don't want to estimate imputation error then set error number and repeats to zero
        error_estimates            = np.array(np.nan)
        error_estimates_percentage = np.array(np.nan)
        original_points            = np.array(np.nan)
        imputed_points             = np.array(np.nan)
    else:
        error_estimates            = np.empty(0)
        error_estimates_percentage = np.empty(0)
        original_points            = np.empty(0)
        imputed_points             = np.empty(0)
    return error_rmse, error_rmse_percentage, fig, ax, error_estimates, error_estimates_percentage, original_points, imputed_points

###############################################################################
###############################################################################
def initialise_outlier_type(outliers: list):
    '''
    Initialisation of the parameters for the outlier tests.

    Parameters
    ----------
    outliers : list
        DESCRIPTION: How to find outliers and the threshold value.
                     Current options are:
                         0) ['nan_only',None]  -- classified all NaN values as outliers/missing data
                         1) ['<',threshold]  -- classifies all values below the threshold as outliers
                         2) ['>',threshold]  -- classifies all values above the threshold as outliers
                         3) ['<>',[low_threshold, hi_threshold]]  -- classifies all values not between the two thresholds as outliers
                         4) ['k',multiplier]  -- classifies all values above the multiplier of the median average deviation as outliers. IMPORTANT NOTE: Does not converge very well/at all if there are consecutive missing values.


    Returns
    -------
    k : float
        DESCRIPTION: multiplier for outlier test
    l_t : float
        DESCRIPTION: lower threshold for outlier test
    g_t : float
        DESCRIPTION: upper threshold for outlier test

    '''
    # Defining the outliers type
    k,l_t,g_t = None,None,None
    if outliers[0] == 'k':
        k = outliers[1]
    elif outliers[0] == '<':
        l_t = outliers[1]
    elif outliers[0] == '>':
        g_t = outliers[1]
    elif outliers[0] == '<>': #here we have to be between two predefined limits
        if not len(outliers[1]) ==2:
            raise ValueError('If input parameter outliers == <> then the second entry in the list should be another length 2 list')
        l_t = outliers[1][0]
        g_t = outliers[1][1]
    elif outliers[0] == 'nan_only':
        pass
    else:
        raise ValueError(f'Outlier type: {outliers[0]} not recognised')
    return k,l_t,g_t

###############################################################################
###############################################################################
def compute_percentage_error(true_vals: np.ndarray, pred_vals: np.ndarray) -> np.ndarray:
    """
    Computes percentage error, normalized against the true ground truth values.
    Returns np.nan where divide-by-zero or infinity occurs.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        err_pct = 100 * (np.abs(true_vals - pred_vals) / np.abs(true_vals))
        err_pct[np.isinf(err_pct)] = np.nan
    return err_pct

###############################################################################
###############################################################################
def initialise_filter(data_per_unit_period: int) -> tuple[np.ndarray, int]:
    '''
    Filter for the case where outliers are determined by a multiplier.
    Not recommended at this stage as it doesn't converge well where there are larger gaps.

    Parameters
    ----------
    data_per_unit_period : int
        DESCRIPTION: How many data points per season period. If season is annual, season_length is number of data points in a year.

    Returns
    -------
    Theta : np.ndarray
        DESCRIPTION: Stationary filter
    ini : int
        DESCRIPTION: Length of the Theta array

    '''
    # % stationary filter
    Theta = np.convolve(np.array([1, -1]),np.concatenate((np.concatenate((np.array([1]), np.zeros(data_per_unit_period-1))), np.array([-1]))))
    ini = len(Theta)
    return Theta, ini
###############################################################################
###############################################################################
def initialise_convergence_error(x_new:       np.ndarray,
                                 out:         np.ndarray,
                                 mu:          float,
                                 convergence: list,
                                 ):
    '''
    Function to initialise convergence error

    Parameters
    ----------
    x_new : np.ndarray
        DESCRIPTION: Input array
    out : np.ndarray
        DESCRIPTION: Array of outliers
    mu : float
        DESCRIPTION: Median of non-outliers
    convergence : list
        DESCRIPTION: How to define the convergence of the outlier fitting method.
                     Current options are:
                         1) ['value', threshold] -- convergence error must be less than the threshold value to signify convergence
                         2) ['min', multiplier]  -- convergence error must be less than the multiplier*(minimum non-outlier value of the data) to signify convergence
                         3) ['med', multiplier]  -- convergence error must be less than the multiplier*(median non-outlier value of the data) to signify convergence

    Returns
    -------
    convergence_error : float
        DESCRIPTION: value that suggests that the iteration has converged.

    '''
    convergence_error = None
    # Defining the errors type IF of type value
    if convergence[0] == 'value':
        convergence_error = convergence[1]
    if convergence[0] == 'min':
        convergence_error = np.abs(convergence[1]*np.min(np.abs(x_new[~out])))  #this take the error as a scalar multiple (usually > 1) of the min value in the (non-outlier) series
    if convergence[0] == 'med':
        convergence_error = np.abs(convergence[1]*mu)

    return convergence_error
###############################################################################
###############################################################################
def find_outliers(x_new:                np.ndarray,
                  outliers:             list,
                  k:                    float,
                  l_t:                  float,
                  g_t:                  float,
                  convergence:          list,
                  data_per_unit_period: int):
    '''
    Function to identify outliers or nan values in a time-series array.
    Additionally, calculates the median, maximum of the non-outlier/nan data, and sets the convergence criteria.

    Parameters
    ----------
    x_new : np.ndarray
        DESCRIPTION: Input array
    outliers : list
        DESCRIPTION: How to find outliers/missing values and the threshold value.
                     Current options are:
                         0) ['nan_only',None]  -- classified all NaN values as outliers/missing data
                         1) ['<',threshold]  -- classifies all values below the threshold as outliers
                         2) ['>',threshold]  -- classifies all values above the threshold as outliers
                         3) ['<>',[low_threshold, hi_threshold]]  -- classifies all values not between the two thresholds as outliers
                         4) ['k',multiplier]  -- classifies all values above the multiplier of the median average deviation as outliers. IMPORTANT NOTE: Does not converge very well/at all if there are consecutive missing values.
    k : float
        DESCRIPTION: multiplier for outlier test
    l_t : float
        DESCRIPTION: lower threshold for outlier test
    g_t : float
        DESCRIPTION: upper threshold for outlier test
    convergence : list
        DESCRIPTION: How to define the convergence of the outlier fitting method.
                     Current options are:
                         1) ['value', threshold] -- convergence error must be less than the threshold value to signify convergence
                         2) ['min', multiplier]  -- convergence error must be less than the multiplier*(minimum non-outlier value of the data) to signify convergence
                         3) ['med', multiplier]  -- convergence error must be less than the multiplier*(median non-outlier value of the data) to signify convergence
    data_per_unit_period : int
        DESCRIPTION: How many data points per season period. If season is annual, season_length is number of data points in a year.

    Returns
    -------
    out : np.ndarray
        DESCRIPTION: Array of True/False values where the True values indicate an outlier/nan/missing value
    mu : float
        DESCRIPTION: median value of all non-outlier data
    mumax : float
        DESCRIPTION: max value of all non-outlier data
    convergence_error : float
        DESCRIPTION: value that suggests that the iteration has converged.

    '''
    from scipy.signal import lfilter
    from scipy import stats
    if outliers[0] == 'k':
        Theta, ini = initialise_filter(data_per_unit_period) # we will add this when/if we need it
        # % Serial Filtering
        if np.nanmin(x_new)>0:
            lx = np.log(x_new)
        else:
            lx = x_new.copy()

        dlx = lfilter(Theta,1,lx.reshape(1,len(lx)))
        se = stats.median_abs_deviation(dlx[0,ini-1:], nan_policy='omit')
        me = np.nanmedian(dlx[0,ini-1:]);

        if np.isnan(se):
            warnings.warn("Warning: Outlier computation window is entirely NaNs. Local detection may fail.")

        y = np.append(me*np.ones((ini-1,1)), dlx[0,ini-1:])
        out = np.abs(y-me)>k*se
    elif outliers[0] == '<':
        out = x_new < l_t
    elif outliers[0] == '>':
        out = x_new > g_t
    elif outliers[0] == '<>':
        out_0 = x_new > g_t
        out_1 = x_new < l_t
        out = np.logical_or(out_0, out_1)
    elif outliers[0] == 'nan_only':
        out = np.isnan(x_new)
    #######################################################################
    # also ALWAYS classify nan values as outliers/missing data
    out = np.logical_or(out,np.isnan(x_new))

    # % values
    mu = np.median(x_new[~out])
    mumax = np.max(x_new[~out])

    #define convergence error
    convergence_error = initialise_convergence_error(x_new,out,mu,convergence) # we will add this later during the iteration

    return out, mu, mumax, convergence_error

###############################################################################
###############################################################################
def remove_good_points_at_random(out:         np.ndarray,
                                 iter_i:      int,
                                 test_number: float,
                                 previous_random_points: np.ndarray = None) -> tuple[np.ndarray,np.ndarray]:
    '''
    Function to randomly select non-outlier points to remove to test the error of the gap filling method.

    Parameters
    ----------
    out : np.ndarray
        DESCRIPTION: Array of True/False values where the True values indicate an outlier/nan/missing value
    iter_i : int
        DESCRIPTION: Current iteration identifier
    test_number : float
        DESCRIPTION: Number of known points to remove in each iteration to help validate the error (the larger the longer the code will take to run but more accurate our error estimate)
    previous_random_points : np.ndarray, optional
        DESCRIPTION: The random points array from the previous iteration.

    Returns
    -------
    new_random_points : np.ndarray
        DESCRIPTION: Array of randomly chosen non-outlier points
    final_out : np.ndarray
        DESCRIPTION: Array of True/False values, including removed non-outlier points, where the True values indicate an outlier/nan/missing value

    '''
    # remove random data points to estimate error.
    # NOTE: This must be done after classifying points as outlier/missing data, otherwise we risk estimating the error using the outliers/missing data points!
    #Note, must also be done after defining max, median etc
    if iter_i == 1 or previous_random_points is None:
        new_random_points = np.zeros_like(out, dtype=bool)
        indices = np.where(~out)[0]
        actual_test_number = min(int(test_number), len(indices))
        selected = np.random.choice(indices, size=actual_test_number, replace=False)
        new_random_points[selected] = True
    else:
        new_random_points = previous_random_points

    if len(out) == 0:
        final_out = out
    else:
        final_out = np.logical_or(out,new_random_points)

    return new_random_points, final_out

###############################################################################
###############################################################################
def initial_guess_for_gap_values(x_new:         np.ndarray,
                                 final_out:     np.ndarray,
                                 initial_guess: list,
                                 mu:         float,
                                 mumax:            float,
                                 use_32_bit: bool  # Added
                                 ) -> np.ndarray:
    '''
    Function to add initial guess/value for outliers/nan/gap data

    Parameters
    ----------
    x_new : np.ndarray
        DESCRIPTION: Input array
    final_out : np.ndarray
        DESCRIPTION: Array of True/False values, including removed non-outlier points, where the True values indicate an outlier/nan/missing value
    initial_guess : list
        DESCRIPTION: How to choose the initial guess for the missing data/outliers.
                     Current options are:
                         1) ['max', ''] -- Initial guess is the maximum of the time series (ignoring outliers)
                         2) ['median', ''] -- Initial guess is the median of the time series
                         3) ['value', numeric] -- Initial guess is the provided numeric value, the second entry in the list.
                         4) ['previous', numeric] -- Initial guess is the previous value of the time series (ignoring any outliers) multiplied by the numeric value, the second entry in the input list.
    mu : float
        DESCRIPTION: median value of all non-outlier data
    mumax : float
        DESCRIPTION: max value of all non-outlier data

    Returns
    -------
    x_new : np.ndarray
        DESCRIPTION: Array with initial guess value added to the gap/nan/outlier values

    '''
    if initial_guess[0] == 'max':
        x_new[final_out] = mumax
    elif initial_guess[0] == 'median':
        x_new[final_out] = mu
    elif initial_guess[0] == 'value':
        x_new[final_out] = initial_guess[1]
    elif initial_guess[0] == 'previous':
        #previous good value used for outlier initial guess
        idx = np.where(~final_out.flatten(), np.arange(len(final_out.flatten())), -1)
        last_good_idx = np.maximum.accumulate(idx)

        bad_indices = np.where(final_out.flatten())[0]

        previous_val = []
        x_new_flat = x_new.flatten()  # Flatten once before the loop to avoid copying overhead
        for outlier_index in bad_indices:
            l_idx = last_good_idx[outlier_index]
            if l_idx != -1:
                previous_val.append(x_new_flat[l_idx].item() * initial_guess[1])
            else:
                previous_val.append(0.0)

        # Reshape previous_val to match final_out's dimension if final_out is 2D
        x_new[final_out] = previous_val

    target_dtype = np.float32 if use_32_bit else np.float64
    try:
        # Attempt to convert the entire array. This assumes that values put into x_new
        # (like from previous_val) are numeric or can be converted.
        x_new = np.asarray(x_new, dtype=target_dtype)
    except ValueError:
        # If conversion fails, it implies some non-numeric data might still be present
        # or was introduced, which should not happen if prior stages cleaned data properly.
        raise ValueError(
            f"Could not convert the initial guess array to the required numeric type ({target_dtype}). "
            "Ensure all data is numeric."
        )

    return x_new

###############################################################################
###############################################################################

def update_imputed_gap_values(x_new: np.ndarray,
                              L: int,
                              extension_type: str,
                              multi_thread_run: bool,
                              component_selection_method: str,
                              number_of_groups_to_drop:   int,
                              eigenvalue_proportion:      float,
                              final_out:                  np.ndarray,
                              use_cissa_overlap:          bool = False,
                              drop_points_from:           str = 'Left',
                              alpha:                      float = 0.05,
                              **kwargs,
                              ):
    '''
    Function which performs a CiSSA step with the most recent data values (i.e. with most recent guess/value for missing/outlier values).
    Selects some components to drop for the update of the missing/outlier values and returns a new timeseries with these removed.

    Parameters
    ----------
    x_new : np.ndarray
        DESCRIPTION: Input time-series array
    L : int
        DESCRIPTION: CiSSA window length.
    extension_type : str
        DESCRIPTION: extension type for left and right ends of the time series.
    multi_thread_run : bool, optional
        DESCRIPTION: Flag to indicate whether the diagonal averaging is performed on multiple cpu cores (True) or not. The default is True.
    component_selection_method : str
        DESCRIPTION: Method for choosing the way we drop components from the reconstruction.
    number_of_groups_to_drop : int
        DESCRIPTION: only used if component_selection_method == 'drop_smallest_n'.
                     Number of components to drop from the reconstruction.
    eigenvalue_proportion : float
        DESCRIPTION: only used if component_selection_method == 'drop_smallest_proportion'.
                     if between 0 and 1, the cumulative proportion psd to keep, or if between -1 and 0, a psd proportion threshold to keep a component.
    final_out : np.ndarray
        DESCRIPTION: Array of True/False values, including removed non-outlier points, where the True values indicate an outlier/nan/missing value
    use_cissa_overlap : bool, optional
        DESCRIPTION. Whether we use ordinary CiSSA (True) or overlap-Cissa (False). The default is False.
    drop_points_from : str, optional
        DESCRIPTION. Only used if use_cissa_overlap == True. If the time series does not divide the overlap exactly, which side to drop data from? The default is 'Left'.
    alpha : float, optional
        DESCRIPTION. Only used if component_selection_method == 'monte_carlo_significant_components'. Significance level for surrogate hypothesis test. For example, --> 100*(1-alpha)% confidence interval. The default is 0.05 (a 95% confidence interval).
    **kwargs : named monte carlo input parameters
    Returns
    -------
    x_new : np.ndarray
        DESCRIPTION: Updated time-series
    x_old : np.ndarray
        DESCRIPTION: pre-updated time-series

    '''
    from pycissa.processing.matrix_operations.matrix_operations import run_cissa
    from pycissa.postprocessing.grouping.grouping_functions import generate_grouping, group
    x_old = x_new.copy()
    if not use_cissa_overlap: #here we simply use OG CiSSA
        Z, psd = run_cissa(x_new,L,extension_type=extension_type,multi_thread_run=multi_thread_run)
        if component_selection_method == 'drop_smallest_n':
            from pycissa.postprocessing.grouping.grouping_functions import drop_smallest_n_components
            temp_array = drop_smallest_n_components(Z,psd,L,number_of_groups_to_drop=number_of_groups_to_drop)
            #
        elif component_selection_method == 'drop_smallest_proportion':
            from pycissa.postprocessing.grouping.grouping_functions import drop_smallest_proportion_psd
            temp_array = drop_smallest_proportion_psd(Z,psd,eigenvalue_proportion)
            #
        elif component_selection_method == 'drop_non_AR_noise (Work in progress)':
            print('NOTE: ADD OPTION HERE TO REMOVE NON-AUTOREGRESSIVE SIGNALS ')
            #
        elif component_selection_method == 'monte_carlo_significant_components':
            warnings.warn("NOTE: The monte_carlo_significant_components method can sometimes be a challenge to converge due to the natural and expected changing number of significant components due to the surrogate testing. If challenges are found, try reducting alpha so that surrogate components rarely switch from non-significant to significant, or alternatively, loosen the convergence tolerance.")
            #
            # If challenges are found, try reducting alpha so that surrogate components rarely switch from non-significant to significant.
            from pycissa.utilities.generate_cissa_result_dictionary import generate_results_dictionary
            from pycissa.postprocessing.grouping.grouping_functions import drop_monte_carlo_non_significant_components
            from pycissa.postprocessing.monte_carlo.montecarlo import run_monte_carlo_test,prepare_monte_carlo_kwargs
            temp_results = generate_results_dictionary(Z,psd,L)
            K_surrogates, surrogates, seed, sided_test, remove_trend,trend_always_significant, A_small_shuffle, generate_toeplitz_matrix = prepare_monte_carlo_kwargs(kwargs)
            from pycissa.postprocessing.grouping.grouping_functions import generate_grouping
            myfrequencies = generate_grouping(psd,L, trend=True)
            temp_result,_ = run_monte_carlo_test(x_new,L,psd,Z,temp_results.get('cissa'),myfrequencies,
                                     alpha=alpha,
                                     K_surrogates=K_surrogates,
                                     surrogates=surrogates,
                                     seed=seed,
                                     sided_test=sided_test,
                                     remove_trend=remove_trend,
                                     trend_always_significant=trend_always_significant,
                                     A_small_shuffle=A_small_shuffle,
                                     extension_type=extension_type,
                                     multi_thread_run=multi_thread_run,
                                     generate_toeplitz_matrix=generate_toeplitz_matrix,
                                     plot_figure=False,
                                     )
            temp_array = drop_monte_carlo_non_significant_components(Z,temp_result,surrogates,alpha)

            #
        else:
            raise ValueError(f"Input parameter component_selection_method was supplied as {component_selection_method}. This MUST be one of 'drop_smallest_n', 'drop_smallest_proportion', or 'drop_non_AR_noise (Work in progress)'.")
        #
        #
    else: #here we use overlap-CiSSA
        from pycissa.processing.cissa.overlap_cissa import OverlapCissa
        # We need to construct Z from kwargs or use a default.
        q = kwargs.get('q', 2 * L)
        L_bar = kwargs.get('L_bar', L // 2)
        Z_len = q + 2 * L_bar

        ocissa = OverlapCissa(t=np.arange(len(x_new)), x=x_new, Z=Z_len, q=q, L=L, use_32_bit=False)
        ocissa.fit(extension_type=extension_type, multi_thread_run=multi_thread_run)

        Z = ocissa.Z
        psd = ocissa.psd

        if component_selection_method == 'drop_smallest_n':
            from pycissa.postprocessing.grouping.grouping_functions import drop_smallest_n_components
            temp_array = drop_smallest_n_components(Z, psd, L, number_of_groups_to_drop=number_of_groups_to_drop)
        elif component_selection_method == 'drop_smallest_proportion':
            from pycissa.postprocessing.grouping.grouping_functions import drop_smallest_proportion_psd
            temp_array = drop_smallest_proportion_psd(Z, psd, eigenvalue_proportion)
        elif component_selection_method == 'monte_carlo_significant_components':
            warnings.warn("NOTE: The monte_carlo_significant_components method can sometimes be a challenge to converge.")
            from pycissa.utilities.generate_cissa_result_dictionary import generate_results_dictionary
            from pycissa.postprocessing.grouping.grouping_functions import drop_monte_carlo_non_significant_components
            from pycissa.postprocessing.monte_carlo.montecarlo import run_monte_carlo_test, prepare_monte_carlo_kwargs
            temp_results = generate_results_dictionary(Z, psd, L)
            K_surrogates, surrogates, seed, sided_test, remove_trend, trend_always_significant, A_small_shuffle, generate_toeplitz_matrix = prepare_monte_carlo_kwargs(kwargs)
            from pycissa.postprocessing.grouping.grouping_functions import generate_grouping
            myfrequencies = generate_grouping(psd, L, trend=True)
            temp_result, _ = run_monte_carlo_test(x_new, L, psd, Z, temp_results.get('cissa'), myfrequencies,
                                     alpha=alpha,
                                     K_surrogates=K_surrogates,
                                     surrogates=surrogates,
                                     seed=seed,
                                     sided_test=sided_test,
                                     remove_trend=remove_trend,
                                     trend_always_significant=trend_always_significant,
                                     A_small_shuffle=A_small_shuffle,
                                     extension_type=extension_type,
                                     multi_thread_run=multi_thread_run,
                                     generate_toeplitz_matrix=generate_toeplitz_matrix,
                                     plot_figure=False)
            temp_array = drop_monte_carlo_non_significant_components(Z, temp_result, surrogates, alpha)
        else:
            raise ValueError(f"Input parameter component_selection_method was supplied as {component_selection_method}. This MUST be one of 'drop_smallest_n', 'drop_smallest_proportion', or 'drop_non_AR_noise (Work in progress)'.")


    updated_values = None

    # updated_values = np.sum(temp_array[out,:],axis=1)
    updated_values = temp_array[final_out]
    x_new[final_out] = updated_values.reshape(x_new[final_out].shape)

    return x_new,x_old,temp_array
###############################################################################
###############################################################################
def produce_error_comparison_figure(original_points:      np.ndarray,
                                    imputed_points:        np.ndarray,
                                    residuals:             np.ndarray,
                                    error_rmse:            float,
                                    error_rmse_percentage: float):
    '''
    Function to plot a figure to check the fit of the gap filling

    Parameters
    ----------
    original_points : np.ndarray
        DESCRIPTION: The original points
    imputed_points : np.ndarray
        DESCRIPTION: The imputation from the gap filling method
    residuals : np.ndarray
        DESCRIPTION: Error residuals
    error_rmse : np.ndarray
        DESCRIPTION: Root mean squared error of the gap filling
    error_rmse_percentage : np.ndarray
        DESCRIPTION: Percentage root mean squared error of the gap filling

    Returns
    -------
    fig : matplotlib.figure
        DESCRIPTION: Figure showing goodness of fit.

    '''
    #plotting one to one figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, constrained_layout=True)
    ax1.plot(original_points, imputed_points, marker = '*',linestyle="None")
    xmin = min(original_points.min(), imputed_points.min())
    xmax = max(original_points.max(), imputed_points.max())
    ax1.plot([xmin, xmax], [xmin, xmax], color="red", linestyle="--")
    ax1.set_title(f"One to one plot. RMSE = {error_rmse}. Percentage RMSE = {error_rmse_percentage}%")
    ax1.set(xlabel='original', ylabel='imputed')

    ax2.plot(original_points, residuals, marker = '*',linestyle="None")
    ax2.plot([xmin, xmax], [0, 0], color="red", linestyle="--")
    ax2.set_title("Residual plot")
    ax2.set(xlabel='original', ylabel='residuals')

    # Histogram of residuals on ax3
    ax3.hist(residuals, bins=30, color='skyblue', edgecolor='black')
    ax3.axvline(0, color='red', linestyle='--', label='Zero Error')
    ax3.set_title("Histogram of Residuals")
    ax3.set(xlabel='Residual', ylabel='Frequency')
    ax3.legend()

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
def plot_time_series_with_imputed_values(t,x_ca,out,rmse,z_value):
    '''
    Function to plot

    Parameters
    ----------
    t : np.ndarray
        DESCRIPTION: Array of input times.
    x_ca : np.ndarray
        DESCRIPTION: Array of data including imputed values.
    out : np.ndarray
        DESCRIPTION: Array of outliers
    rmse : float
        DESCRIPTION: Root mean squared error of the imputation
    z_value : float
        DESCRIPTION: z-value (= 1.96 for a 95% confidence interval, for example)

    Returns
    -------
    fig : matplotlib.figure
        DESCRIPTION: Figure on time series including imputed outliers

    '''
    fig, ax = plt.subplots()

    try:
        x_ca = x_ca.reshape(len(x_ca),)
    except:
        raise ValueError(f'"x_ca" should be a column vector. The size of x_ca is ({x_ca.shape})')

    try:
        out = out.reshape(len(out),)
    except:
        raise ValueError(f'"out" should be a column vector. The size of out is ({out.shape})')

    # print(out)
    ax.plot(t[~out], x_ca[~out], 'b', lw=1.0, label = 'original series')
    if sum(out)>0:
        if not np.isnan(z_value*rmse):
            ax.errorbar(t[out],  x_ca[out],  fmt = 'ro',yerr = [z_value*rmse]*len(x_ca[out]), lw=1.0, label = 'imputed points')
        else:
            ax.plot(t[out], x_ca[out], 'r+', lw=1.0, label = 'imputed points')
    else: warnings.warn("WARNING: No gaps found in the data.")

    fig.legend(loc="upper left")

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

###############################################################################
###############################################################################
def fill_timeseries_gaps_iterative_components(t:                          np.ndarray,
                         x:                          np.ndarray,
                         L:                          int,
                         convergence:                list = ['value', 1],
                         extension_type:             str  = 'AR_LR',
                         multi_thread_run:           bool = True,
                         initial_guess:              list = ['previous', 1],
                         outliers:                   list = ['nan_only',None],
                         estimate_error:             bool  = True,
                         test_number:                int = 10,
                         test_repeats:               int = 5,
                         z_value:                    float = 1.96,
                         component_selection_method: str = 'drop_smallest_n',
                         min_number_of_groups_to_drop:int = 1,
                         data_per_unit_period:       int = 1,
                         use_cissa_overlap:          bool = False,
                         drop_points_from:           str = 'Left',
                         max_iter:                   int = 100,
                         verbose:                    bool = False,
                          use_32_bit:                 bool = False,
                         **kwargs,
                         ):
    '''
    Function to fill in gaps (NaN values) and/or replace outliers in a timeseries via imputation.
    This is achieved by replacing gaps/outliers with an initial guess and then iteratively running the CiSSA (or overlap-CiSSA) method, keeping some (but not all) of the reconstructed series in each step of the algorithm, until convergence is achieved.
    Optionally, we evaluate the accuracy of the imputation by testing known points.

    -------------------------------------------------------------------------
    References:
    [1] Bógalo, J., Poncela, P., & Senra, E. (2021).
        "Circulant singular spectrum analysis: a new automated procedure for signal extraction".
          Signal Processing, 179, 107824.
        https://doi.org/10.1016/j.sigpro.2020.107824.
    [2] Bógalo, J., Llada, M., Poncela, P., & Senra, E. (2022).
        "Seasonality in COVID-19 times".
          Economics Letters, 211, 110206.
          https://doi.org/10.1016/j.econlet.2021.110206
    -------------------------------------------------------------------------

    Parameters
    ----------
    t : np.ndarray
        DESCRIPTION: Array of input times.
    x : np.ndarray
        DESCRIPTION: Input time series array which possibly has
    L : int
        DESCRIPTION: CiSSA window length.
    convergence : list, optional
        DESCRIPTION. How to define the convergence of the outlier fitting method.
                     Current options are:
                         1) ['value', threshold] -- convergence error must be less than the threshold value to signify convergence
                         2) ['min', multiplier]  -- convergence error must be less than the multiplier*(minimum non-outlier value of the data) to signify convergence
                         3) ['med', multiplier]  -- convergence error must be less than the multiplier*(median non-outlier value of the data) to signify convergence.
                    The default is ['value', 1].
    extension_type : str, optional
        DESCRIPTION. extension type for left and right ends of the time series. The default is 'AR_LR'.
    multi_thread_run : bool, optional
        DESCRIPTION. Flag to indicate whether the diagonal averaging is performed on multiple cpu cores (True) or not. The default is True.. The default is True.
    initial_guess : list, optional
        DESCRIPTION. How to choose the initial guess for the missing data/outliers.
                     Current options are:
                         1) ['max', ''] -- Initial guess is the maximum of the time series (ignoring outliers)
                         2) ['median', ''] -- Initial guess is the median of the time series
                         3) ['value', numeric] -- Initial guess is the provided numeric value, the second entry in the list.
                         4) ['previous', numeric] -- Initial guess is the previous value of the time series (ignoring any outliers) multiplied by the numeric value, the second entry in the input list.
                    The default is ['previous', 1].
    outliers : list, optional
        DESCRIPTION. How to find outliers/missing values and the threshold value.
                     Current options are:
                         0) ['nan_only',None]  -- classified all NaN values as outliers/missing data
                         1) ['<',threshold]  -- classifies all values below the threshold as outliers
                         2) ['>',threshold]  -- classifies all values above the threshold as outliers
                         3) ['<>',[low_threshold, hi_threshold]]  -- classifies all values not between the two thresholds as outliers
                         4) ['k',multiplier]  -- classifies all values above the multiplier of the median average deviation as outliers. IMPORTANT NOTE: Does not converge very well/at all if there are consecutive missing values.
                    The default is ['nan_only',None].
    estimate_error : bool, optional
        DESCRIPTION: Flag which determines if we will be estimating the error in the gap filling or not. The default is True.
    test_number : int
        DESCRIPTION: Number of known points to remove in each iteration to help validate the error (the larger the longer the code will take to run but more accurate our error estimate). The default is 10.
    test_repeats : int
        DESCRIPTION: Number of times to repeat the gap filling process to estimate error (the larger the longer the code will take to run but more accurate our error estimate). The default is 1.
    z_value : float, optional
        DESCRIPTION: z-value for confidence interval (= 1.96 for a 95% confidence interval, for example)
    component_selection_method : str, optional
        DESCRIPTION. Method for choosing the way we drop components from the reconstruction. Current options are 'drop_smallest_n', 'drop_smallest_proportion', 'monte_carlo_significant_components'. The default is 'monte_carlo_significant_components'.
    eigenvalue_proportion : float, optional
        DESCRIPTION. only used if component_selection_method == 'drop_smallest_proportion'.
                     if between 0 and 1, the cumulative proportion psd to keep, or if between -1 and 0, a psd proportion threshold to keep a component.
                     The default is 0.95.
    number_of_groups_to_drop : int, optional
        DESCRIPTION. only used if component_selection_method == 'drop_smallest_n'.
                     Number of components to drop from the reconstruction.
                     The default is 1.
    data_per_unit_period : int, optional
        DESCRIPTION. How many data points per season period. If season is annual, season_length is number of data points in a year.
                     The default is 1.
    use_cissa_overlap : bool, optional
        DESCRIPTION. Whether we use ordinary CiSSA (True) or overlap-Cissa (False). The default is False. The default is False.
    drop_points_from : str, optional
        DESCRIPTION. Only used if use_cissa_overlap == True. If the time series does not divide the overlap exactly, which side to drop data from. The default is 'Left'.
    max_iter : int, optional
        DESCRIPTION. Maximum number of iterations to check for convergence. The default is 50.
    verbose : bool, optional
        DESCRIPTION. Whether to print some info to the console or not. The default is False.

    Returns
    -------
    x_ca : np.ndarray
        DESCRIPTION: Array with gaps (possibly) filled.
    error_estimates : np.ndarray|None
        DESCRIPTION: Array of errors associated with the test points
    error_estimates_percentage : np.ndarray|None
        DESCRIPTION: Array of percentage errors associated with the test points.
    error_rmse : float|None
        DESCRIPTION: Root mean squared error of test points.
    error_rmse_percentage : float|None
        DESCRIPTION: Percentage root mean squared error of test points.
    original_points : np.ndarray|None
        DESCRIPTION: Array of the original time series points
    imputed_points : np.ndarray|None
        DESCRIPTION: Array of the imputed time series points.
    fig_errors : matplotlit.figure|None
        DESCRIPTION: Figure plotting the error metrics (accuracy of the gap filling imputation)
    fig_time_series : matplotlit.figure|None
        DESCRIPTION: Figure plotting the time series with imputed values.
    '''
    #ensure we don't get rid of more than half the time series... may need to make this even more strict in the future.
    if test_number > (len(x)/2):
        test_number = int(np.floor((len(x)/2)))

    number_of_cissa_components = int(np.floor(L/2))

    if test_repeats == 0:
        test_repeats = 1 #need at least one to quantify the error
    if not estimate_error:
        estimate_error=True #need to quantify the error


    # 1. Validate inputs
    x =  validate_input_parameters(x,L,extension_type,outliers,initial_guess,convergence)

    # 2. Initialise some variables
    error_rmse, error_rmse_percentage, fig, ax, error_estimates, error_estimates_percentage, original_points, imputed_points = initialise_error_estimates(estimate_error)
    k,l_t,g_t = initialise_outlier_type(outliers)


    optimising_error_temp= {}
    optimising_percentage_error_temp= {}
    optimising_original_points_temp= {}
    optimising_imputed_points_temp= {}
    # 3. Begin outlier/missing data iterations
    for testrepeats in range(0,test_repeats + 1):
        mulist = []
        if verbose: print(f'Step {testrepeats} of {test_repeats}')
        if testrepeats == test_repeats:
            test_number = 0
            def compute_rms(arrays):
                merged = np.concatenate(arrays)
                return np.sqrt(np.mean(merged ** 2))
            min_key = min(optimising_error_temp, key=lambda k: compute_rms(optimising_error_temp[k]))
            min_number_of_groups_to_drop = min_key-1
            if verbose:
                print("Optimal number of components to drop: ",min_key)


        # 3a. initial allocation
        x_old   = x.copy()
        x_new   = x.copy()
        x_ca    = x.copy()
        x_final = x.copy()*0
        for num_groups_to_drop in range(int(number_of_cissa_components)-1,min_number_of_groups_to_drop,-1):
            if verbose: print(f'   Dropping {num_groups_to_drop} of {number_of_cissa_components}')

            x_new = x-x_final
            x_old = x_new.copy()
            # 3b. initialise iteration number and ensure if this is the last iteration we DO NOT remove any test points

            iter_i = 1

            # 3c. run through while loop.
            out, mu, mumax, convergence_error = find_outliers(x_new,outliers,k,l_t,g_t,convergence,data_per_unit_period)
            # print(mu,mumax,convergence_error)
            mulist.append(f'{num_groups_to_drop}:{mu}-{mumax}')
            if sum(out) == 0:
                warnings.warn("WARNING: No gaps found in the data. Returning the original (unmodified) time-series.")
                return x,None,None,None,None,None,None,None,None

            # 3c-i. Find outliers/missing data and convergence criterion
            if num_groups_to_drop == int(number_of_cissa_components)-1:
                x_final[out] = 0
                # 3c-ii. Randomly select non-outlier points to evaluate error in gap filling
                new_random_points, final_out  = remove_good_points_at_random(out,iter_i,test_number)



            # 3c-iii. Add initial guess to outlier/nan/ gap points
            x_new = initial_guess_for_gap_values(x_new,final_out,initial_guess,mu,mumax,use_32_bit)

            # 3c-iv. Iterate through
            current_error = 1.1*convergence_error
            while_iter = 0
            while current_error>convergence_error:
                x_new,x_old,temp_array = update_imputed_gap_values(x_new,L,extension_type,multi_thread_run,component_selection_method,
                                                        num_groups_to_drop,None,final_out,use_cissa_overlap=use_cissa_overlap,drop_points_from=drop_points_from,
                                                        alpha=None,
                                                        K_surrogates=None,
                                                        surrogates=None,
                                                        seed=None,
                                                        sided_test=None,
                                                        remove_trend=None,
                                                        trend_always_significant=None,
                                                        A_small_shuffle=None,
                                                        generate_toeplitz_matrix=False)
                current_error = np.max(np.abs(x_old-x_new))
                # print(x_old[400],x_new[400],np.max(np.abs(x_old-x_new)),np.max(np.abs(x_old[final_out]-x_new[final_out])))
                # print(x_old[final_out],x_new[final_out])
                if verbose: print(f'iteration {while_iter}. ',current_error,' vs target error: ',convergence_error)
                while_iter += 1
                if while_iter > max_iter:
                    raise GapFillConvergenceError(f"Exceeded max number of iterations ({max_iter}) without convergence")

            x_final += temp_array

            if verbose: print(f'END iteration: {iter_i}, error: {np.max(np.abs(x_ca-x_new))} vs target error: {convergence_error}')

            # % corrected series
            x_ca = x_final.copy()
            error_estimates_temp            = np.append(error_estimates,np.abs(x[new_random_points] - x_ca[new_random_points]))
            err_pct_temp                    = compute_percentage_error(x[new_random_points], x_ca[new_random_points])
            error_estimates_percentage_temp = np.append(error_estimates_percentage, err_pct_temp)
            original_points_temp            = np.append(original_points,x[new_random_points])
            imputed_points_temp             = np.append(imputed_points,x_ca[new_random_points])
            optimising_error_temp.setdefault(num_groups_to_drop, []).append(error_estimates_temp)
            optimising_percentage_error_temp.setdefault(num_groups_to_drop, []).append(error_estimates_percentage_temp)
            optimising_original_points_temp.setdefault(num_groups_to_drop, []).append(original_points_temp)
            optimising_imputed_points_temp.setdefault(num_groups_to_drop, []).append(imputed_points_temp)


        #4. Update error estimation and points.
    error_estimates            = np.concatenate(optimising_error_temp.get(min_key))
    error_estimates_percentage = np.concatenate(optimising_percentage_error_temp.get(min_key))
    original_points            = np.concatenate(optimising_original_points_temp.get(min_key))
    imputed_points             = np.concatenate(optimising_imputed_points_temp.get(min_key))


    #5. Calculate RMSE and residuals
    error_rmse             = np.sqrt( (np.sum(error_estimates*error_estimates))/len(error_estimates)    )
    error_rmse_percentage  = np.sqrt( (np.sum(error_estimates_percentage*error_estimates_percentage))/len(error_estimates_percentage)    )
    residuals = original_points - imputed_points

    #TO DO. INVESTIGATE CONFORMAL PREDICTION METHODS FOR ADDING PREDICTION INTERVALS

    #6 create figures
    if estimate_error:
        if len(imputed_points) > 0:
            fig_errors = produce_error_comparison_figure(original_points,imputed_points,residuals,error_rmse,error_rmse_percentage)
        else:
            fig_errors = None
            warnings.warn("WARNING: No gaps found in the data.")
    else: fig_errors = None

    #7 Make a figure here which plots the original time series and also the imputed values.
    fig_time_series = plot_time_series_with_imputed_values(t,x_ca,out,error_rmse,z_value)

    return x_ca,error_estimates,error_estimates_percentage,error_rmse,error_rmse_percentage,original_points,imputed_points, fig_errors,fig_time_series

###############################################################################
###############################################################################
def fill_timeseries_gaps(t:                          np.ndarray,
                         x:                          np.ndarray,
                         L:                          int,
                         convergence:                list = ['value', 1],
                         extension_type:             str  = 'AR_LR',
                         multi_thread_run:           bool = True,
                         initial_guess:              list = ['previous', 1],
                         outliers:                   list = ['nan_only',None],
                         estimate_error:             bool  = True,
                         test_number:                int = 10,
                         test_repeats:               int = 5,
                         z_value:                    float = 1.96,
                         component_selection_method: str = 'monte_carlo_significant_components',
                         eigenvalue_proportion:      float = 0.95,
                         number_of_groups_to_drop:   int = 1,
                         data_per_unit_period:       int = 1,
                         use_cissa_overlap:          bool = False,
                         drop_points_from:           str = 'Left',
                         max_iter:                   int = 50,
                         verbose:                    bool = False,
                         alpha:                      float = 0.05,
                          use_32_bit:                 bool = False,
                         **kwargs,
                         ):
    '''
    Function to fill in gaps (NaN values) and/or replace outliers in a timeseries via imputation.
    This is achieved by replacing gaps/outliers with an initial guess and then iteratively running the CiSSA (or overlap-CiSSA) method, keeping some (but not all) of the reconstructed series in each step of the algorithm, until convergence is achieved.
    Optionally, we evaluate the accuracy of the imputation by testing known points.

    -------------------------------------------------------------------------
    References:
    [1] Bógalo, J., Poncela, P., & Senra, E. (2021).
        "Circulant singular spectrum analysis: a new automated procedure for signal extraction".
          Signal Processing, 179, 107824.
        https://doi.org/10.1016/j.sigpro.2020.107824.
    [2] Bógalo, J., Llada, M., Poncela, P., & Senra, E. (2022).
        "Seasonality in COVID-19 times".
          Economics Letters, 211, 110206.
          https://doi.org/10.1016/j.econlet.2021.110206
    -------------------------------------------------------------------------

    Parameters
    ----------
    t : np.ndarray
        DESCRIPTION: Array of input times.
    x : np.ndarray
        DESCRIPTION: Input time series array which possibly has
    L : int
        DESCRIPTION: CiSSA window length.
    convergence : list, optional
        DESCRIPTION. How to define the convergence of the outlier fitting method.
                     Current options are:
                         1) ['value', threshold] -- convergence error must be less than the threshold value to signify convergence
                         2) ['min', multiplier]  -- convergence error must be less than the multiplier*(minimum non-outlier value of the data) to signify convergence
                         3) ['med', multiplier]  -- convergence error must be less than the multiplier*(median non-outlier value of the data) to signify convergence.
                    The default is ['value', 1].
    extension_type : str, optional
        DESCRIPTION. extension type for left and right ends of the time series. The default is 'AR_LR'.
    multi_thread_run : bool, optional
        DESCRIPTION. Flag to indicate whether the diagonal averaging is performed on multiple cpu cores (True) or not. The default is True.. The default is True.
    initial_guess : list, optional
        DESCRIPTION. How to choose the initial guess for the missing data/outliers.
                     Current options are:
                         1) ['max', ''] -- Initial guess is the maximum of the time series (ignoring outliers)
                         2) ['median', ''] -- Initial guess is the median of the time series
                         3) ['value', numeric] -- Initial guess is the provided numeric value, the second entry in the list.
                         4) ['previous', numeric] -- Initial guess is the previous value of the time series (ignoring any outliers) multiplied by the numeric value, the second entry in the input list.
                    The default is ['previous', 1].
    outliers : list, optional
        DESCRIPTION. How to find outliers/missing values and the threshold value.
                     Current options are:
                         0) ['nan_only',None]  -- classified all NaN values as outliers/missing data
                         1) ['<',threshold]  -- classifies all values below the threshold as outliers
                         2) ['>',threshold]  -- classifies all values above the threshold as outliers
                         3) ['<>',[low_threshold, hi_threshold]]  -- classifies all values not between the two thresholds as outliers
                         4) ['k',multiplier]  -- classifies all values above the multiplier of the median average deviation as outliers. IMPORTANT NOTE: Does not converge very well/at all if there are consecutive missing values.
                    The default is ['nan_only',None].
    estimate_error : bool, optional
        DESCRIPTION: Flag which determines if we will be estimating the error in the gap filling or not. The default is True.
    test_number : int
        DESCRIPTION: Number of known points to remove in each iteration to help validate the error (the larger the longer the code will take to run but more accurate our error estimate). The default is 10.
    test_repeats : int
        DESCRIPTION: Number of times to repeat the gap filling process to estimate error (the larger the longer the code will take to run but more accurate our error estimate). The default is 1.
    z_value : float, optional
        DESCRIPTION: z-value for confidence interval (= 1.96 for a 95% confidence interval, for example)
    component_selection_method : str, optional
        DESCRIPTION. Method for choosing the way we drop components from the reconstruction. Current options are 'drop_smallest_n', 'drop_smallest_proportion', 'monte_carlo_significant_components'. The default is 'monte_carlo_significant_components'.
    eigenvalue_proportion : float, optional
        DESCRIPTION. only used if component_selection_method == 'drop_smallest_proportion'.
                     if between 0 and 1, the cumulative proportion psd to keep, or if between -1 and 0, a psd proportion threshold to keep a component.
                     The default is 0.95.
    number_of_groups_to_drop : int, optional
        DESCRIPTION. only used if component_selection_method == 'drop_smallest_n'.
                     Number of components to drop from the reconstruction.
                     The default is 1.
    data_per_unit_period : int, optional
        DESCRIPTION. How many data points per season period. If season is annual, season_length is number of data points in a year.
                     The default is 1.
    use_cissa_overlap : bool, optional
        DESCRIPTION. Whether we use ordinary CiSSA (True) or overlap-Cissa (False). The default is False. The default is False.
    drop_points_from : str, optional
        DESCRIPTION. Only used if use_cissa_overlap == True. If the time series does not divide the overlap exactly, which side to drop data from. The default is 'Left'.
    max_iter : int, optional
        DESCRIPTION. Maximum number of iterations to check for convergence. The default is 50.
    verbose : bool, optional
        DESCRIPTION. Whether to print some info to the console or not. The default is False.

    Returns
    -------
    x_ca : np.ndarray
        DESCRIPTION: Array with gaps (possibly) filled.
    error_estimates : np.ndarray|None
        DESCRIPTION: Array of errors associated with the test points
    error_estimates_percentage : np.ndarray|None
        DESCRIPTION: Array of percentage errors associated with the test points.
    error_rmse : float|None
        DESCRIPTION: Root mean squared error of test points.
    error_rmse_percentage : float|None
        DESCRIPTION: Percentage root mean squared error of test points.
    original_points : np.ndarray|None
        DESCRIPTION: Array of the original time series points
    imputed_points : np.ndarray|None
        DESCRIPTION: Array of the imputed time series points.
    fig_errors : matplotlit.figure|None
        DESCRIPTION: Figure plotting the error metrics (accuracy of the gap filling imputation)
    fig_time_series : matplotlit.figure|None
        DESCRIPTION: Figure plotting the time series with imputed values.
    '''
    #ensure we don't get rid of more than half the time series... may need to make this even more strict in the future.
    if test_number > (len(x)/2):
        test_number = int(np.floor((len(x)/2)))

    from pycissa.postprocessing.monte_carlo.montecarlo import prepare_monte_carlo_kwargs
    K_surrogates, surrogates, seed, sided_test, remove_trend,trend_always_significant, A_small_shuffle, generate_toeplitz_matrix = prepare_monte_carlo_kwargs(kwargs)


    # 1. Validate inputs
    x =  validate_input_parameters(x,L,extension_type,outliers,initial_guess,convergence)

    # 2. Initialise some variables
    error_rmse, error_rmse_percentage, fig, ax, error_estimates, error_estimates_percentage, original_points, imputed_points = initialise_error_estimates(estimate_error)
    k,l_t,g_t = initialise_outlier_type(outliers)

    # 3. Begin outlier/missing data iterations
    for testrepeats in range(0,test_repeats + 1):
        if verbose: print(f'Step {testrepeats} of {test_repeats}')
        # 3a. initial allocation
        x_old = x.copy()
        x_new = x.copy()
        x_ca = x.copy()

        # 3b. initialise iteration number and ensure if this is the last iteration we DO NOT remove any test points
        if testrepeats == test_repeats:
            test_number = 0
        iter_i = 1
        new_random_points = None
        imputation_started = False

        # 3c. run through while loop.
        while iter_i>0:
            # 3c-i. Find outliers/missing data and convergence criterion
            out, mu, mumax, convergence_error = find_outliers(x_new,outliers,k,l_t,g_t,convergence,data_per_unit_period)
            if sum(out) == 0:
                if not imputation_started:
                    warnings.warn("WARNING: No gaps found in the data. Returning the original (unmodified) time-series.")
                    return x,None,None,None,None,None,None,None,None
                else:
                    break
            imputation_started = True
            # 3c-ii. Randomly select non-outlier points to evaluate error in gap filling
            new_random_points, final_out  = remove_good_points_at_random(out,iter_i,test_number,new_random_points)

            # 3c-iii. Add initial guess to outlier/nan/ gap points
            x_new = initial_guess_for_gap_values(x_new,final_out,initial_guess,mu,mumax,use_32_bit)

            # 3c-iv. Iterate through
            current_error = 1.1*convergence_error
            while_iter = 0
            while current_error>convergence_error:
                x_new,x_old,temp_array = update_imputed_gap_values(x_new,L,extension_type,multi_thread_run,component_selection_method,
                                                        number_of_groups_to_drop,eigenvalue_proportion,final_out,use_cissa_overlap=use_cissa_overlap,drop_points_from=drop_points_from,
                                                        alpha=alpha,
                                                        K_surrogates=K_surrogates,
                                                        surrogates=surrogates,
                                                        seed=seed,
                                                        sided_test=sided_test,
                                                        remove_trend=remove_trend,
                                                        trend_always_significant=trend_always_significant,
                                                        A_small_shuffle=A_small_shuffle,
                                                        generate_toeplitz_matrix=generate_toeplitz_matrix)
                current_error = np.max(np.abs(x_old-x_new))
                if verbose: print(f'iteration {while_iter}. ',current_error,' vs target error: ',convergence_error)
                while_iter += 1
                if while_iter > max_iter:
                    raise GapFillConvergenceError(f"Exceeded max number of iterations ({max_iter}) without convergence")


            # 3c-v. Check convergence
            if np.nanmax(np.abs(x_ca-x_new))>convergence_error:
                iter_i += 1
            else:
                x_ca = x_new.copy()
                iter_i = 0

            # % corrected series
            x_ca = x_new.copy()


            if iter_i > max_iter:
                raise GapFillConvergenceError(f"Exceeded max number of iterations ({max_iter}) without convergence")

        #4. Update error estimation and points.
        error_estimates            = np.append(error_estimates,np.abs(x[new_random_points] - x_ca[new_random_points]))
        err_pct                    = compute_percentage_error(x[new_random_points], x_ca[new_random_points])
        error_estimates_percentage = np.append(error_estimates_percentage, err_pct)
        original_points            = np.append(original_points,x[new_random_points])
        imputed_points             = np.append(imputed_points,x_ca[new_random_points])

    #5. Calculate RMSE and residuals
    if len(error_estimates) > 0:
        error_rmse             = np.sqrt( np.nanmean(error_estimates*error_estimates) )
        error_rmse_percentage  = np.sqrt( np.nanmean(error_estimates_percentage*error_estimates_percentage) )
    else:
        error_rmse = np.nan
        error_rmse_percentage = np.nan
        if estimate_error:
            warnings.warn("WARNING: No test points were held out; error metrics unavailable.")
    residuals = original_points - imputed_points

    #TO DO. INVESTIGATE CONFORMAL PREDICTION METHODS FOR ADDING PREDICTION INTERVALS

    #6 create figures
    if estimate_error:
        if len(imputed_points) > 0:
            fig_errors = produce_error_comparison_figure(original_points,imputed_points,residuals,error_rmse,error_rmse_percentage)
        else:
            fig_errors = None
            warnings.warn("WARNING: No gaps found in the data.")
    else: fig_errors = None

    #7 Make a figure here which plots the original time series and also the imputed values.
    fig_time_series = plot_time_series_with_imputed_values(t,x_ca,out,error_rmse,z_value)

    return x_ca,error_estimates,error_estimates_percentage,error_rmse,error_rmse_percentage,original_points,imputed_points, fig_errors,fig_time_series


import numpy as np

class GapFillConvergenceError(Exception):
    """Exception raised when gap filling algorithms fail to converge within max_iter."""
    pass

def update_imputed_m_gap_values(x_new: np.ndarray, L: int, extension_type: str, multi_thread_run: bool, component_selection_method: str, number_of_groups_to_drop: int, eigenvalue_proportion: float, final_out: np.ndarray, use_cissa_overlap: bool=False, drop_points_from: str='Left', alpha: float=0.05, **kwargs):
    from pycissa.processing.matrix_operations.m_matrix_operations import run_mcissa
    import warnings
    x_old = x_new.copy()

    # We must only extend if we're not overlapping, but gap fill assumes extending.
    T_orig = x_new.shape[0]
    Z_stacked, psd, Zs = run_mcissa(x_new, L, extension_type=extension_type, extend_left=True, extend_right=True)

    # run_mcissa ALREADY slices down to T in m_group_paired_frequencies!
    # Z_stacked has shape (T, M, nft) out of run_mcissa
    T, M, nft = Z_stacked.shape

    temp_array = np.zeros((T_orig, M))
    if component_selection_method == 'drop_smallest_n':
        variances = np.sum([np.var(Z_stacked[:, m, :], axis=0) for m in range(M)], axis=0)
        sorted_indices = np.argsort(variances)[::-1] # descending
        keep_indices = sorted_indices[:-number_of_groups_to_drop] if number_of_groups_to_drop > 0 else sorted_indices
        for i in keep_indices:
            temp_array += Z_stacked[:, :, i]
    elif component_selection_method == 'drop_smallest_proportion':
        variances = np.sum([np.var(Z_stacked[:, m, :], axis=0) for m in range(M)], axis=0)
        sorted_indices = np.argsort(variances)[::-1]
        cumulative_prop = np.cumsum(variances[sorted_indices]) / np.sum(variances)
        keep_indices = []
        for idx, prop in zip(sorted_indices, cumulative_prop):
            keep_indices.append(idx)
            if prop >= eigenvalue_proportion:
                break
        for i in keep_indices:
            temp_array += Z_stacked[:, :, i]
    elif component_selection_method == 'monte_carlo_significant_components':
        warnings.warn("NOTE: The monte_carlo_significant_components method can sometimes be a challenge to converge due to the natural and expected changing number of significant components due to the surrogate testing. If challenges are found, try reducting alpha so that surrogate components rarely switch from non-significant to significant, or alternatively, loosen the convergence tolerance.")
        from pycissa.postprocessing.grouping.m_grouping_functions import m_drop_monte_carlo_non_significant_components

        # Merge extension_type into kwargs to pass down correctly
        if 'extension_type' not in kwargs:
            kwargs['extension_type'] = extension_type
        if 'alpha' not in kwargs:
            kwargs['alpha'] = alpha

        temp_array = m_drop_monte_carlo_non_significant_components(
            Z_stacked=Z_stacked,
            psd=psd,
            L=L,
            x_new=x_new,
            **kwargs
        )
    else:
        # Simplest fallback
        temp_array = np.sum(Z_stacked, axis=2)

    updated_values = temp_array[final_out]
    x_new[final_out] = updated_values.reshape(x_new[final_out].shape)
    return x_new, x_old, temp_array

def m_fill_timeseries_gaps(t, x, L, convergence=['value', 1], extension_type='AR_LR', multi_thread_run=True, initial_guess=['previous', 1], outliers=['nan_only', None], estimate_error=True, test_number=10, test_repeats=5, z_value=1.96, component_selection_method='drop_smallest_n', eigenvalue_proportion=0.95, number_of_groups_to_drop=1, data_per_unit_period=1, max_iter=50, verbose=False, **kwargs):
    from pycissa.preprocessing.gap_fill.gap_filling import initialise_error_estimates, initialise_outlier_type, find_outliers, remove_good_points_at_random, initial_guess_for_gap_values, produce_error_comparison_figure, plot_time_series_with_imputed_values
    import warnings

    # 1. Init
    error_rmse, error_rmse_percentage, fig, ax, error_estimates, error_estimates_percentage, original_points, imputed_points = initialise_error_estimates(estimate_error)
    k, l_t, g_t = initialise_outlier_type(outliers)

    # ensure we don't get rid of more than half the time series
    if test_number > (len(x)/2):
        test_number = int(np.floor((len(x)/2)))

    # M-channel support
    M = x.shape[1]
    use_32_bit = kwargs.get('use_32_bit', False)

    # 3. Begin outlier/missing data iterations
    for testrepeats in range(0, test_repeats + 1):
        if verbose: print(f'Step {testrepeats} of {test_repeats}')
        x_old = x.copy()
        x_new = x.copy()
        x_ca = x.copy()

        if testrepeats == test_repeats:
            test_number = 0

        iter_i = 1
        new_random_points_mask_full = None
        imputation_started = False

        while iter_i > 0:
            out_mask = np.zeros(x_new.shape, dtype=bool)
            mu_m = np.zeros(M)
            mumax_m = np.zeros(M)
            conv_err_m = np.zeros(M)

            for m in range(M):
                out_1d, mu_1d, mumax_1d, conv_1d = find_outliers(x_new[:, m], outliers, k, l_t, g_t, convergence, data_per_unit_period)
                out_mask[:, m] = out_1d
                mu_m[m] = mu_1d
                mumax_m[m] = mumax_1d
                conv_err_m[m] = conv_1d

            if np.sum(out_mask) == 0:
                if not imputation_started:
                    warnings.warn("WARNING: No gaps found in the data. Returning the original (unmodified) time-series.")
                    return x, None, None, None, None, None, None, None, None
                else:
                    break
            imputation_started = True

            convergence_error = np.abs(np.nanmax(conv_err_m))

            final_out_mask = np.zeros(x_new.shape, dtype=bool)
            new_random_points_mask = np.zeros(x_new.shape, dtype=bool)

            for m in range(M):
                prev_mask_m = new_random_points_mask_full[:, m] if new_random_points_mask_full is not None else None
                nrp, f_o = remove_good_points_at_random(out_mask[:, m], iter_i, test_number, prev_mask_m)
                new_random_points_mask[:, m] = nrp
                final_out_mask[:, m] = f_o

                # add initial guess
                x_new_m = initial_guess_for_gap_values(x_new[:, m], f_o, initial_guess, mu_m[m], mumax_m[m], use_32_bit)
                x_new[:, m] = x_new_m

            new_random_points_mask_full = new_random_points_mask.copy()

            current_error = convergence_error + 1.0
            while_iter = 0

            while current_error > convergence_error:
                x_new, x_old, temp_array = update_imputed_m_gap_values(
                    x_new, L, extension_type, multi_thread_run, component_selection_method,
                    number_of_groups_to_drop, eigenvalue_proportion, final_out_mask, **kwargs
                )
                current_error = np.max(np.abs(x_old - x_new))
                if verbose: print(f'iteration {while_iter}. ', current_error, ' vs target error: ', convergence_error)
                while_iter += 1
                if while_iter > max_iter:
                    raise GapFillConvergenceError(f"Exceeded max number of iterations ({max_iter}) without convergence")

            if np.nanmax(np.abs(x_ca - x_new)) > convergence_error:
                iter_i += 1
            else:
                x_ca = x_new.copy()
                iter_i = 0

            x_ca = x_new.copy()

            if iter_i > max_iter:
                raise GapFillConvergenceError(f"Exceeded max number of iterations ({max_iter}) without convergence")

        if test_number > 0:
            error_estimates = np.append(error_estimates, np.abs(x[new_random_points_mask] - x_ca[new_random_points_mask]))
            err_pct = compute_percentage_error(x[new_random_points_mask], x_ca[new_random_points_mask])
            error_estimates_percentage = np.append(error_estimates_percentage, err_pct)
            original_points = np.append(original_points, x[new_random_points_mask])
            imputed_points = np.append(imputed_points, x_ca[new_random_points_mask])

    # 5. Calculate RMSE and residuals
    if len(error_estimates) > 0:
        error_rmse = np.sqrt(np.nanmean(error_estimates * error_estimates))
        error_rmse_percentage = np.sqrt(np.nanmean(error_estimates_percentage * error_estimates_percentage))
    else:
        error_rmse = np.nan
        error_rmse_percentage = np.nan
        if estimate_error:
            warnings.warn("WARNING: No test points were held out; error metrics unavailable.")
    residuals = original_points - imputed_points

    # 6 create figures
    if estimate_error and len(imputed_points) > 0:
        fig_errors = produce_error_comparison_figure(original_points, imputed_points, residuals, error_rmse, error_rmse_percentage)
    else:
        fig_errors = None

    fig_time_series = []
    # For simplicity, we just plot one channel or multi if needed. We'll just return None for multivariate or the first channel
    fig_time_series = plot_time_series_with_imputed_values(t, x_ca[:, 0], out_mask[:, 0], error_rmse, z_value)

    return x_ca, error_estimates, error_estimates_percentage, error_rmse, error_rmse_percentage, original_points, imputed_points, fig_errors, fig_time_series