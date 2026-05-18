import numpy as np
import warnings
import matplotlib.pyplot as plt


def initial_data_checks(t: np.ndarray, x: np.ndarray, use_32_bit: bool):
    '''
    Data checks to ensure t,x are numpy arrays of the correct shape.
    Will try to convert to the correct shape if they are not

    Parameters
    ----------
    t : np.ndarray
        DESCRIPTION: Array of input times.
    x : np.ndarray
        DESCRIPTION: Array of input data.
    use_32_bit : bool
        DESCRIPTION: Flag to indicate whether to convert x to float32.

    Raises
    ------
    ValueError
        DESCRIPTION: Exception raised if the input variables are either not arrays or can't be converted to an array, or if the array shape is incorrect.'

    Returns
    -------
    t : np.ndarray
        DESCRIPTION: Array of (possible reshaped) input times.
    x : np.ndarray
        DESCRIPTION: Array of (possible reshaped) input data.
    '''
    ######################################
    #check x is a numpy array
    if not type(x) is np.ndarray:
        try:
            x = np.array(x)
            x = x.reshape(len(x),)
        except: raise ValueError(f'Input "x" is not a numpy array, nor can be converted to one.')
    myshape = x.shape
    if not len(myshape) == 1:
        try:
            x = x.reshape(len(x),)
        except:
            raise ValueError(f'Input "x" should be a column vector (i.e. only contain a single column). The size of x is ({myshape})')

    if use_32_bit:
        new_x = np.empty_like(x, dtype=object)
        for i, val in enumerate(x):
            try:
                new_x[i] = np.float32(val)
            except (ValueError, OverflowError):
                new_x[i] = val
        x = new_x
        # Try to convert the entire array to float32 if all elements are numbers,
        # otherwise leave as object type to accommodate mixed types.
        try:
            x = x.astype(np.float32)
        except ValueError:
            pass # x remains with dtype=object

    ######################################
    #check t is a numpy array
    if not type(t) is np.ndarray:
        try:
            t = np.array(t)
            t = t.reshape(len(t),)
        except: raise ValueError(f'Input "t" is not a numpy array, nor can be converted to one.')
    myshape = t.shape

    if not len(myshape) == 1:
        try:
            t = t.reshape(len(t),)
        except:
            raise ValueError(f'Input "t" should be a column vector (i.e. only contain a single column). The size of t is ({myshape})')


    # Type Check for t
    unique_types = set(type(item) for item in t)
    if len(unique_types) > 1:
        is_all_numeric = all(issubclass(typ, (int, float, np.integer, np.floating)) for typ in unique_types)
        if not is_all_numeric:
            raise TypeError(f"Time vector t contains mixed data types: {list(unique_types)}")


    # Handle non-numeric t arrays and datetime64 conversion
    if not all(isinstance(item, (int, float, np.integer, np.floating)) for item in t):
        if t.dtype != np.dtype('datetime64[ns]'):
            try:
                t = t.astype('datetime64[ns]')
            except ValueError:
                if len(t) > 1:
                    raise ValueError("Time vector t could not be converted to datetime. Please ensure it is in a format convertible to datetime64.")
        if t.dtype == np.dtype('datetime64[ns]'):
            # Check if sorted in ascending order
            if not all(t[i] <= t[i+1] for i in range(len(t)-1)):
                # Find the first unsorted element for a more informative message
                for i in range(len(t) - 1):
                    if not t[i+1] >= t[i]:
                        raise ValueError(f"Time vector t is not sorted in ascending order. Element {t[i+1]} at index {i+1} is less than preceding element {t[i]} at index {i}.")

    # Order Check for t
    is_numeric = all(isinstance(item, (int, float, np.integer, np.floating)) for item in t)
    if is_numeric and len(t) > 1:
        for i in range(len(t) - 1):
            if not t[i+1] >= t[i]:
                raise ValueError(f"Time vector t is not sorted in ascending order. Element {t[i+1]} at index {i+1} is less than preceding element {t[i]} at index {i}.")
    return t,x


class Cissa:
    '''
    Circulant Singular Spectrum Analysis: Data must be equally spaced!
    '''
    def __init__(self, t: np.ndarray, x: np.ndarray, use_32_bit: bool = False):
        #----------------------------------------------------------------------
        self.use_32_bit = use_32_bit # Assign attribute first
        # perform initial checks to ensure input variables are numpy arrays of the correct shape.
        t,x = initial_data_checks(t,x,self.use_32_bit)
        self.x_raw = x #array of corresponding data
        self.t_raw = t #array of corresponding data
        #----------------------------------------------------------------------

        #----------------------------------------------------------------------
        #perform check for censored data
        from pycissa.preprocessing.data_cleaning.data_cleaning import detect_censored_data
        self.censored,num_censored = detect_censored_data(x)
        if self.censored:
            warnings.warn("WARNING: Censored data detected. Please run pre_fix_censored_data before fitting.")
            self.information_text += f'''
            ------------------------------------------------------
            {num_censored} censored data points found.
            '''
        #----------------------------------------------------------------------

        #----------------------------------------------------------------------
        #perform check for nan data
        from pycissa.preprocessing.data_cleaning.data_cleaning import detect_nan_data
        self.isnan = detect_nan_data(x)
        if self.isnan: warnings.warn("WARNING: nan data detected. Please run pre_fill_gaps before fitting.")
        #----------------------------------------------------------------------

        #----------------------------------------------------------------------
        self.t = t #array of times
        self.x = x #array of corresponding data

        self.information_text = ''  #information about outputs

        if not hasattr(self, 'figures'):
            self.figures = {}  #make a space for future figures
        self.figures.update({'cissa':{}})
        #----------------------------------------------------------------------
    def restore_original_data(self):
        '''
        Method to restore original data (x,t) = (x_raw,t_raw)
        '''
        from pycissa.preprocessing.data_cleaning.data_cleaning import detect_censored_data,detect_nan_data
        self.x = self.x_raw
        self.t = self.t_raw
        self.censored,num_censored = detect_censored_data(self.x)  #if we restore the data we must check if the restored data is censored again...
        self.isnan = detect_nan_data(self.x)
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def fit(self,
            L: int,
            extension_type: str = 'AR_LR',
            multi_thread_run: bool = True,
            num_workers: int = 2,
            generate_toeplitz_matrix: bool = False,
            extend_left: bool = True,
            extend_right: bool = True):
        '''
        Function to fit CiSSA to a timeseries.
        -------------------------------------------------------------------------
        References:
        [1] Bógalo, J., Poncela, P., & Senra, E. (2021).
            "Circulant singular spectrum analysis: a new automated procedure for signal extraction".
              Signal Processing, 179, 107824.
            https://doi.org/10.1016/j.sigpro.2020.107824.
        -------------------------------------------------------------------------

        Parameters
        ----------
        x : np.ndarray
            DESCRIPTION: Input array
        L : int
            DESCRIPTION: CiSSA window length.
        extension_type : str
            DESCRIPTION: extension type for left and right ends of the time series. The default is AR_LR
                         Options are:
                             'Mirror' - data at the start and end of the series is mirrored, x_{L+1} = X_{L-1},
                             'NoExt' - no extension applied (Not recommended...),
                             'AR_LR' - autoregressive extension applied to start (L) and end (R) of x,
                             'AR_L' - autoregressive extension applied to start (L) of x, ,
                             'AR_R - autoregressive extension applied to end (R) of x'
        multi_thread_run : bool, optional
            DESCRIPTION: Flag to indicate whether the diagonal averaging is performed on multiple threads (True) or not. The default is True.
        num_workers : int, optional
            DESCRIPTION: If using multi-threading, how many workers to use.
        generate_toeplitz_matrix : bool, optional
            DESCRIPTION: Flag to indicate whether we need to calculate the symmetric Toeplitz matrix or not. The default is False.

        Returns
        -------
        Z : np.ndarray
            DESCRIPTION: Output CiSSA results.
        psd : np.ndarray
            DESCRIPTION: estimation of the the circulant matrix power spectral density

        '''
        #----------------------------------------------------------------------
        #ensure data is not uncensored or nan
        if self.censored:  raise ValueError("Censored data detected. Please run pre_fix_censored_data before fitting.")
        if self.isnan: raise ValueError("WARNING: nan data detected. Please run pre_fill_gaps before fitting.")
        #----------------------------------------------------------------------

        # Determine target dtype and convert self.x
        target_dtype = np.float32 if self.use_32_bit else np.float64
        try:
            self.x = np.asarray(self.x, dtype=target_dtype)
        except ValueError:
            raise ValueError("All elements in the input array 'x' must be numeric or convertible to numeric type before fitting. Please check for non-numeric values.")

        #run cissa
        from pycissa.processing.matrix_operations.matrix_operations import run_cissa
        self.Z, self.psd = run_cissa(self.x,
                                      L,
                                      extension_type=extension_type,
                                      multi_thread_run=multi_thread_run,
                                      num_workers=num_workers,
                                      generate_toeplitz_matrix=generate_toeplitz_matrix,
                                      extend_left=extend_left,
                                      extend_right=extend_right)

        #generate initial results dictionary
        from pycissa.utilities.generate_cissa_result_dictionary import generate_results_dictionary
        if not hasattr(self, 'results'):
            self.results = generate_results_dictionary(self.Z,self.psd,L)
        else:
            self.results.update(generate_results_dictionary(self.Z,self.psd,L))

        from pycissa.postprocessing.grouping.grouping_functions import generate_grouping
        myfrequencies = generate_grouping(self.psd,L, trend=True)
        self.frequencies = myfrequencies

        results = self.results
        results.get('cissa').setdefault('model parameters', {})
        results.get('cissa').setdefault('noise component tests', {})
        results.get('cissa').setdefault('fractal scaling results', {})
        results.get('cissa').get('model parameters').update({
            'extension_type'   : extension_type,
            'L'                : L,
            'multi_thread_run' : multi_thread_run,
            })
        self.results = results


        #save settings
        self.extension_type = extension_type
        self.L = L
        self.multi_thread_run = multi_thread_run

        return self
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def predict(self):
        print("FUTURE PREDICTION NOT YET IMPLEMENTED")
        #TO DO, maybe using AutoTS or MAPIE?
        '''
        Joining CI (trend, periodic, noise)
        https://arxiv.org/pdf/2406.16766
        '''
        return self

    def plot_original_time_series(self):
        '''
        Helper function to plot the original time series
        '''
        from pycissa.utilities.plotting import plot_time_series
        #----------------------------------------------------------------------
        #ensure data is not uncensored or nan
        if self.censored:  raise ValueError("Censored data detected. Please run pre_fix_censored_data before plot_original_time_series.")
        if self.isnan: raise ValueError("WARNING: nan data detected. Please run pre_fill_gaps before plot_original_time_series.")
        #----------------------------------------------------------------------
        fig = plot_time_series(self.t,self.x)
        self.figures['cissa'].update({'figure_original_time_series':fig})
        if plt.get_fignums(): plt.close('all')
        return self
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def pre_fill_gaps(self,
                  L:                          int,
                  convergence:                list|None = None,
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
                  min_number_of_groups_to_drop:int = 1,
                  data_per_unit_period:       int = 1,
                  use_cissa_overlap:          bool = False,
                  drop_points_from:           str = 'Left',
                  max_iter:                   int = 100,
                  verbose:                    bool = False,
                  alpha:                      float = 0.05,
                  **kwargs,
                  ):
        '''
        Function to fill in gaps (NaN values) and/or replace outliers in a timeseries via imputation.
        This is achieved by replacing gaps/outliers with an initial guess and then iteratively running the CiSSA (or overlap-CiSSA) method, keeping some (but not all) of the reconstructed series in each step of the algorithm, until convergence is achieved.
        Optionally, we evaluate the accuracy of the imputation by testing known points (HIGHLY RECOMMENDED).

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
                        The default is ['value', 0.01 * np.nanmin(self.x)].
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
            DESCRIPTION. Flag which determines if we will be estimating the error in the gap filling or not. The default is True.
        test_number : int
            DESCRIPTION: Number of known points to remove in each iteration to help validate the error (the larger the longer the code will take to run but more accurate our error estimate). The default is 10.
        test_repeats : int
            DESCRIPTION: Number of times to repeat the gap filling process to estimate error (the larger the longer the code will take to run but more accurate our error estimate). The default is 5.
        z_value : float, optional
            DESCRIPTION: z-value for confidence interval (= 1.96 for a 95% confidence interval, for example)
        component_selection_method : str, optional
            DESCRIPTION. Method for choosing the way we drop components from the reconstruction. Current options are 'drop_smallest_n', 'drop_smallest_proportion', 'monte_carlo_significant_components', or to perform iterative imputation (iterating through all components from largest to smallest, adding one more at a time), select 'add_components_iteratively'. The default is 'monte_carlo_significant_components'.

        eigenvalue_proportion : float, optional
            DESCRIPTION. only used if component_selection_method == 'drop_smallest_proportion'.
                         if between 0 and 1, the cumulative proportion psd to keep, or if between -1 and 0, a psd proportion threshold to keep a component.
                         The default is 0.95.
        number_of_groups_to_drop : int, optional
            DESCRIPTION. only used if component_selection_method == 'add_components_iteratively'.
                         Number of components to drop from the reconstruction.
                         The default is 1.
        min_number_of_groups_to_drop  : int, optional
            DESCRIPTION. only used if component_selection_method == 'drop_smallest_n'.
                         MINIMUM number of components to drop from the reconstruction during the optimisation.
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
        alpha : float, optional
            DESCRIPTION. Only used if component_selection_method == 'monte_carlo_significant_components'. Significance level for surrogate hypothesis test. For example, --> 100*(1-alpha)% confidence interval. The default is 0.05 (a 95% confidence interval).
        **kwargs : named monte carlo input parameters

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

        if convergence is None:
            convergence = ['value', 0.01 * np.nanmin(self.x)]

        if component_selection_method == 'add_components_iteratively':
            from pycissa.preprocessing.gap_fill.gap_filling import fill_timeseries_gaps_iterative_components
            x_ca,error_estimates,error_estimates_percentage,error_rmse,error_rmse_percentage,original_points,imputed_points, fig_errors,fig_time_series = fill_timeseries_gaps_iterative_components(
                                    self.t,
                                    self.x,
                                     L,
                                     convergence=convergence,
                                     extension_type=extension_type,
                                     multi_thread_run=multi_thread_run,
                                     initial_guess=initial_guess,
                                     outliers=outliers,
                                     estimate_error=estimate_error,
                                     number_of_groups_to_drop=number_of_groups_to_drop,
                                     test_number=test_number,
                                     test_repeats=test_repeats,
                                     z_value=z_value,
                                     data_per_unit_period=data_per_unit_period,
                                     use_cissa_overlap=use_cissa_overlap,
                                     drop_points_from=drop_points_from,
                                     max_iter=max_iter,
                                     verbose=verbose,
                                     use_32_bit=self.use_32_bit,
                                     )
        else:
            from pycissa.preprocessing.gap_fill.gap_filling import fill_timeseries_gaps
            from pycissa.postprocessing.monte_carlo.montecarlo import prepare_monte_carlo_kwargs
            if self.censored:  raise ValueError("Censored data detected. Please run pre_fix_censored_data before fitting.")
            K_surrogates, surrogates, seed, sided_test, remove_trend,trend_always_significant, A_small_shuffle, generate_toeplitz_matrix = prepare_monte_carlo_kwargs(kwargs)
            x_ca,error_estimates,error_estimates_percentage,error_rmse,error_rmse_percentage,original_points,imputed_points, fig_errors,fig_time_series = fill_timeseries_gaps(
                                    self.t,
                                    self.x,
                                     L,
                                     convergence=convergence,
                                     extension_type=extension_type,
                                     multi_thread_run=multi_thread_run,
                                     initial_guess=initial_guess,
                                     outliers=outliers,
                                     estimate_error=estimate_error,
                                     z_value=z_value,
                                     component_selection_method=component_selection_method,
                                     eigenvalue_proportion=eigenvalue_proportion,
                                     number_of_groups_to_drop=number_of_groups_to_drop,
                                     data_per_unit_period=data_per_unit_period,
                                     use_cissa_overlap=use_cissa_overlap,
                                     drop_points_from=drop_points_from,
                                     max_iter=max_iter,
                                     test_number=test_number,
                                     test_repeats=test_repeats,
                                     verbose=verbose,
                                     alpha=alpha,
                                     K_surrogates=K_surrogates,
                                     surrogates=surrogates,
                                     seed=seed,
                                     sided_test=sided_test,
                                     remove_trend=remove_trend,
                                     trend_always_significant=trend_always_significant,
                                     A_small_shuffle=A_small_shuffle,
                                     generate_toeplitz_matrix=generate_toeplitz_matrix,
                                     use_32_bit=self.use_32_bit,)

        self.x = x_ca.reshape(len(x_ca),)
        self.gap_fill_error_estimates            = error_estimates
        self.gap_fill_error_estimates_percentage = error_estimates_percentage
        self.gap_fill_error_rmse                 = error_rmse
        self.gap_fill_error_rmse_percentage      = error_rmse_percentage
        self.gap_fill_original_points            = original_points
        self.gap_fill_imputed_points             = imputed_points,
        # self.figure_gap_fill_error               = fig_errors,
        # self.figure_gap_fill                     = fig_time_series
        self.figures.get('cissa').update({'figure_gap_fill_error':fig_errors,
                            'figure_gap_fill'      :fig_time_series,
                            })
        from pycissa.preprocessing.data_cleaning.data_cleaning import detect_nan_data
        self.isnan = detect_nan_data(self.x)
        if plt.get_fignums(): plt.close('all')

        self.information_text += f'''
        ------------------------------------------------------
        gap fill RMSE  : {self.gap_fill_error_rmse}
        gap fill % RMSE: {self.gap_fill_error_rmse_percentage}
        '''#information about outputs

        return self
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def pre_fill_uneven_timeseries(self,
                                   L_values: list[int],
                                   dt: float,
                                   gap_threshold: float,
                                   eps_values: list[float] = None,
                                   interp_method: str = 'cubic',
                                   optimization_metric: str = 'rmse',
                                   r2_warning_threshold: float = 0.5,
                                   plot: bool = True,
                                   center_data: bool = False,
                                   **kwargs):
        """
        Wrapper for fill_uneven_timeseries that optionally updates the Cissa object's data to the even grid.

        Parameters
        ----------
        L_values : list[int]
            List of L (window length) parameters to optimize over for CiSSA gap filling.
        dt : float
            Grid spacing for the evenly sampled grid.
        gap_threshold : float
            Max distance to a real data point on the even grid before it is considered a gap (NaN).
        eps_values : list[float], optional
            List of convergence epsilon values to optimize over. If None, defaults to [1.0].
        interp_method : str, optional
            Interpolation method. The default is 'cubic'.
        optimization_metric : str, optional
            Metric to optimize when selecting L ('rmse' or 'ccc'). The default is 'rmse'.
        r2_warning_threshold : float, optional
            R-squared threshold below which a warning is issued. Default is 0.5.
        plot : bool, optional
            Whether to produce diagnostic plots.
        center_data : bool, optional
            If True, replaces self.t and self.x with the generated evenly sampled and filled grid.
        **kwargs :
            Additional arguments passed to fill_timeseries_gaps.

        Returns
        -------
        self : Cissa
        """
        from pycissa.preprocessing.gap_fill.uneven_gap_filling import fill_uneven_timeseries

        # We pass self.t and self.x which might be uneven
        results = fill_uneven_timeseries(t=self.t,
                                         x=self.x,
                                         L_values=L_values,
                                         dt=dt,
                                         gap_threshold=gap_threshold,
                                         eps_values=eps_values,
                                         interp_method=interp_method,
                                         optimization_metric=optimization_metric,
                                         r2_warning_threshold=r2_warning_threshold,
                                         plot=plot,
                                         **kwargs)

        # Optionally, save the figure and results in the cissa object
        if 'fig' in results:
            self.figures.get('cissa').update({'figure_uneven_gap_fill': results['fig']})
            if plt.get_fignums(): plt.close('all')

        self.uneven_gap_fill_results = results

        if center_data:
            self.t = results['t_even']
            self.x = results['x_even_filled']

            # Re-check for nan data on the new grid
            from pycissa.preprocessing.data_cleaning.data_cleaning import detect_nan_data
            self.isnan = detect_nan_data(self.x)

            self.information_text += f'''
        ------------------------------------------------------
        Data centered to evenly sampled grid.
        '''

        return self

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def pre_fix_censored_data(self,
                             replace_type:        str = 'raw',
                             lower_multiplier:    float = 0.5,
                             upper_multiplier:    float = 1.1,
                             default_value_lower: float = 0.,
                             default_value_upper: float = 0.,
                             hicensor_lower:      bool = False,
                             hicensor_upper:      bool = False,
                             ):
        '''
        Function to find and replace upper and lower censored data in an input array x.
        There are three different types of replacemet for censored values:

        replacement_type = 'raw'      -> Here, censored data is simply replaced by the numeric value. e.g. '<1' becomes 1
        replacement_type = 'multiple' -> Here, censored data is replaced by a multiple multipled the censored data numeric value. For example, '<1' becomes 1* lower_multiplier, while '>1' becomes 1*upper_multiplier.
        replacement_type = 'constant' -> Here, censored data is replaced by a constant value as defined by the input variables default_value_lower and default_value_upper.

        Additionally, the function has the option to apply hicensoring to censored data. This is an option where all of the lower censored values are replaced with the largest lower censored value after processing (e.g. after applying a multiplier). Similarly, all of the upper censored values can be replaced with the smallest upper censored value after processing.
        This is useful for data that has multiple levels of censoring to help avoid bias and reduce the potential for trends apprearing only because of a changing censoring level.
        See, for example, Helsel, D. R. (1990). Less than obvious-statistical treatment of data below the detection limit. Environmental science & technology, 24(12), 1766-1774.


        NOTE 1: Any entries that are not numeric nor censored are converted to np.nan
        NOTE 2: In the future we would like to impute values for censored data based on assumed empirical distribution. See for example, https://cran.r-project.org/web/packages/NADA2/index.html

        Parameters
        ----------
        x : np.ndarray
            DESCRIPTION: array of input data.
        replacement_type : str, optional
            DESCRIPTION: Type of replacememt if a censored value is found. Allowed values are 'raw', 'multiple', or 'constant'. The default is 'raw'.
        lower_multiplier : float, optional
            DESCRIPTION. Only used if replacememt_type == 'multiple'. This is the multiplier to apply to a lower censored data point. For example, a point '<1' will become '1*lower_multiplier'. The default is 0.5.
        upper_multiplier : float, optional
            DESCRIPTION. Only used if replacememt_type == 'multiple'. This is the multiplier to apply to a upper censored data point. For example, a point '>1' will become '1*upper_multiplier'. The default is 1.1.
        default_value_lower : float, optional
            DESCRIPTION. Only used if replacement_type == 'constant'. The numeric value to replace any left (lower) censored data. For example, '<1' becomes 'default_value_lower'. The default is 0.-
        default_value_upper : float, optional
            DESCRIPTION. Only used if replacement_type == 'constant'. The numeric value to replace any right (upper) censored data. For example, '<1' becomes 'default_value_upper'. The default is 0.
        hicensor_lower : bool, optional
            DESCRIPTION. Whether lower censored data should be replaced with the largest (replaced) censored value. The default is False.
        hicensor_upper : bool, optional
            DESCRIPTION. Whether upper censored data should be replaced with the smallest (replaced) censored value. The default is False.


        Returns
        -------
        x_uncensored : np.ndarray
            DESCRIPTION: array of now uncensored data
        x_censoring : np.ndarray
            DESCRIPTION: array of locations where any censoring was found. Value of the array = None if no censoring is found at a given array position.

        '''
        if self.censored:
            from pycissa.preprocessing.data_cleaning.data_cleaning import _fix_censored_data, detect_nan_data, detect_censored_data
            self.x,self.censoring = _fix_censored_data(self.x,
                                     replacement_type = replace_type,
                                     lower_multiplier = lower_multiplier,
                                     upper_multiplier = upper_multiplier,
                                     default_value_lower = default_value_lower,
                                     default_value_upper = default_value_upper,
                                     hicensor_lower = hicensor_lower,
                                     hicensor_upper = hicensor_upper,)
            self.isnan = detect_nan_data(self.x)
            self.censored,_ = detect_censored_data(self.x)
            self.information_text += f'''
            ------------------------------------------------------
            Censored data replaced
            '''

        else: warnings.warn("WARNING: No censored data detected. Returning unchanged data.")

        return self
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    from datetime import datetime
    def pre_fix_missing_samples(
            self,
            version:              str = 'date',
            start_date:           str|datetime = 'min',
            date_settings:        dict = {'input_dateformat'  :'',
                                          'years'             :0,
                                          'months'            :1,
                                          'days'              :0,
                                          'hours'             :0,
                                          'minutes'           :0,
                                          'seconds'           :0,
                                          'year_delta'        :0,
                                          'month_delta'       :0,
                                          'day_delta'         :14,
                                          'hour_delta'        :0,
                                          'minute_delta'      :0,
                                          'second_delta'      :0,
                                          },
            numeric_time_settings: dict = {'t_step'    :1.,
                                           'wiggleroom':0.99
                                            },
            missing_value:      int = np.nan
            ):
        '''
        Function that finds and corrects missing values in the time series.
        Missing dates result in adding a default value "missing_value" into the input data.

        **THIS FUNCTION IS A WORK IN PROGRESS. USE WITH EXTREME CAUTION.**

        Parameters
        ----------
        self : Cissa object
            DESCRIPTION: Cissa object
        version : str, optional
            DESCRIPTION: String describing the type of time data. One of 'date' or 'numeric'. The default is 'date'.
        start_date : str|datetime
            DESCRIPTION: Only used if version = 'date'. If start_date = 'min' then the minimum date is used, otherwise the given datetime is taken as the first required time. The default is 'min'.
        date_settings : dict, optional
            DESCRIPTION: Dictionary of date settings as defined below:
                                {
                                years : int, optional
                                    DESCRIPTION: (ideal) number of years between each timestep in input array t. The default is 0.
                                months : int, optional
                                    DESCRIPTION: (ideal) number of months between each timestep in input array t. The default is 1.
                                days : int, optional
                                    DESCRIPTION: (ideal) number of days between each timestep in input array t. The default is 0.
                                hours : int, optional
                                    DESCRIPTION: (ideal) number of hours between each timestep in input array t. The default is 0.
                                minutes : int, optional
                                    DESCRIPTION: (ideal) number of minutes between each timestep in input array t. The default is 0.
                                seconds : int, optional
                                    DESCRIPTION: (ideal) number of seconds between each timestep in input array t. The default is 0.
                                input_dateformat : str, optional
                                    DESCRIPTION: Datetime string format. The default is '%Y'. See https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes
                                year_delta : int, optional
                                    DESCRIPTION: Integer years to build a tolerance interval around the desired timestep. If the time is within the "wiggleroom", then the time is OK. For example, if we have a monthly sampling frequency on the 15th of the month, but one sample is on the 14th, we don't want to say that the sample is missing. The default is 0.
                                month_delta : int, optional
                                    DESCRIPTION: Integer months to build a tolerance interval around the desired timestep. If the time is within the "wiggleroom", then the time is OK. For example, if we have a monthly sampling frequency on the 15th of the month, but one sample is on the 14th, we don't want to say that the sample is missing. The default is 0.
                                day_delta : int, optional
                                    DESCRIPTION: Integer days to build a tolerance interval around the desired timestep. If the time is within the "wiggleroom", then the time is OK. For example, if we have a monthly sampling frequency on the 15th of the month, but one sample is on the 14th, we don't want to say that the sample is missing. The default is 0.
                                hour_delta : int, optional
                                    DESCRIPTION: Integer hours to build a tolerance interval around the desired timestep. If the time is within the "wiggleroom", then the time is OK. For example, if we have a monthly sampling frequency on the 15th of the month, but one sample is on the 14th, we don't want to say that the sample is missing. The default is 0.
                                minute_delta : int, optional
                                    DESCRIPTION: Integer minutes to build a tolerance interval around the desired timestep. If the time is within the "wiggleroom", then the time is OK. For example, if we have a monthly sampling frequency on the 15th of the month, but one sample is on the 14th, we don't want to say that the sample is missing. The default is 0.
                                second_delta : int, optional
                                    DESCRIPTION: Integer seconds to build a tolerance interval around the desired timestep. If the time is within the "wiggleroom", then the time is OK. For example, if we have a monthly sampling frequency on the 15th of the month, but one sample is on the 14th, we don't want to say that the sample is missing. The default is 0.
                                    }
        numeric_time_settings : dict, optional
            Dictionary of date settings as defined below:
                               {
                               t_step : int|float, optional
                                   DESCRIPTION: numeric value of the time step. The default is 1.
                               wiggleroom : int|float, optional
                                   DESCRIPTION: Numeric value for the 'wiggle room' associated with a tolerance tolerance interval around the desired timestep. If the time is within the "wiggleroom", then the time is OK. For example, if we have a time step of 2 and the wiggle room is 0.2, then a series of times 2,4,6,7.9,10,... would be OK, but 2,4,6,7.7,10 would not and would correct the time value to 2,4,6,8,10. The default is 0.99.
                                   }
        missing_value : int, optional
            DESCRIPTION: The value which is entered when a missing value is found. The default is np.nan.

        Returns
        -------
        final_t : np.ndarray
            DESCRIPTION: array of corrected time values (i.e. missing values are added)
        final_x : np.ndarray
            DESCRIPTION: array of corrected data values (i.e. missing values are added)
        x_missing : np.ndarray
            DESCRIPTION: array of values indicating whether a value is added or not. If not, None, if so, the value will be True.

        '''

        if version == 'date':
            #1) check that the needed settings are present
            if not (date_settings.get('years',0)+date_settings.get('months',0)+date_settings.get('days',0)+date_settings.get('hours',0)+date_settings.get('minutes',0)+date_settings.get('seconds',0)) > 0: raise ValueError(f"At least one date step must be provided and greater than zero. Please check the 'years', 'months', 'days', 'hours', 'minutes', and 'seconds' in date_settings (Note, some of these may be excluded or zero, but at least one should be provided and >0 )")
            from pycissa.preprocessing.data_cleaning.data_cleaning import _fix_missing_date_samples
            self.t,self.x,self.added_times, self.t_centered = _fix_missing_date_samples(
                                     self.t,
                                     self.x,
                                     start_date,
                                       years             = date_settings.get('years',0),
                                       months            = date_settings.get('months',0),
                                       days              = date_settings.get('days',0),
                                       hours             = date_settings.get('hours',0),
                                       minutes           = date_settings.get('minutes',0),
                                       seconds           = date_settings.get('seconds',0),
                                       input_dateformat  = date_settings.get('input_dateformat',0),
                                       year_delta        = date_settings.get('year_delta',0),
                                       month_delta       = date_settings.get('month_delta',0),
                                       day_delta         = date_settings.get('day_delta',0),
                                       hour_delta        = date_settings.get('hour_delta',0),
                                       minute_delta      = date_settings.get('minute_delta',0),
                                       second_delta      = date_settings.get('second_delta',0),
                                       missing_value     = missing_value)

        elif version == 'numeric':
            from pycissa.preprocessing.data_cleaning.data_cleaning import _fix_missing_numeric_samples
            self.t,self.x,self.added_times,self.t_centered = _fix_missing_numeric_samples(
                                        self.t,
                                        self.x,
                                       t_step         = numeric_time_settings.get('t_step',1),
                                       wiggleroom     = numeric_time_settings.get('wiggleroom',0.99),
                                       missing_value  = missing_value
                                       )
        else: raise ValueError(f"Input parameter 'version' shpuld be one of 'date' or 'numeric', depending on the time data type. You entered: {version}.")


        from pycissa.preprocessing.data_cleaning.data_cleaning import detect_nan_data
        self.isnan = detect_nan_data(self.x)

        self.information_text += f'''
        ------------------------------------------------------
        {self.added_times} number of samples missing in the time series to ensure it is approximately evenly spaced.
        '''

        return self

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def post_run_frequency_time_analysis(self,
                                    data_per_period:    int,
                                    period_name:        str = '',
                                    t_unit:             str = '',
                                    plot_frequency:     bool = True,
                                    plot_period:        bool = True,
                                    logplot_frequency:  bool = True,
                                    logplot_period:     bool = False,
                                    normalise_plots:    bool = False,
                                    height_variable:    str = 'power',
                                    height_unit:        str = '',):
        '''
        Function to generate frequency-time and period-time matrices and figures.

        Parameters
        ----------
        Z : np.ndarray
            DESCRIPTION: Output CiSSA results.
        psd : np.ndarray
            DESCRIPTION: estimation of the the circulant matrix power spectral density
        t : np.ndarray
            DESCRIPTION: Array of input times.
        L : int
            DESCRIPTION: CiSSA window length.
        data_per_period : int
            DESCRIPTION: Number of data points/time steps in a user-defined period (for example, for monthly data and a user-desired period of years, data_per_period = 12)
        period_name : str, optional
            DESCRIPTION: Names of the user-defined period (e.g. years, months, hours etc). The default is ''.
        t_unit : str, optional
            DESCRIPTION: Time unit (can also be generic such as 'date'). The default is ''.
        plot_frequency : bool, optional
            DESCRIPTION: Flag, whether to produce a frequency-time plot or not. The default is True.
        plot_period : bool, optional
            DESCRIPTION: Flag, whether to produce a period-time plot or not. The default is False.
        logplot_frequency : bool, optional
            DESCRIPTION: Flag, whether to plot the frequency-time plot on a log-scale or not. The default is False.
        logplot_period : bool, optional
            DESCRIPTION: Flag, whether to plot the period-time plot on a log-scale or not. The default is False.
        normalise_plots : bool, optional
            DESCRIPTION: Flag, whether to normalise the plots or not. The default is False.
        height_variable : str, optional
            DESCRIPTION: The height variable to plot. One of 'power', 'amplitude', or 'phase'. The default is 'power'.
        height_unit : str, optional
            DESCRIPTION: Unit of the height variable. The default is ''.

        Returns
        -------
        freq_list : list
            DESCRIPTION: List of frequencies of the signals obtained via CiSSA
        period_list : list
            DESCRIPTION: List of periods of the signals obtained via CiSSA
        amplitude_matrix : np.ndarray
            DESCRIPTION: Array of amplitudes - split via time and for each frequency
        power_matrix : np.ndarray
            DESCRIPTION: Array of power - split via time and for each frequency
        phase_matrix : np.ndarray
            DESCRIPTION: Array of phases - split via time and for each frequency
        frequency_matrix : np.ndarray
            DESCRIPTION: Array of Hilbert Frequencies - split via time and for each frequency
        fig_f : matplotlib.figure
            DESCRIPTION: Frequency-time figure
        fig_p : matplotlib.figure
            DESCRIPTION: Period-time figure

        '''
        from pycissa.postprocessing.frequency_time.frequency_time import _run_frequency_time_analysis

        #check that all necessary input variables exist
        necessary_attributes = ["Z","psd","t","L","results"]
        for attr_i in necessary_attributes:
            if not hasattr(self, attr_i): raise ValueError(f"Attribute {attr_i} does not appear to exist in the class. Please fun the pycissa fit method before running the run_frequency_time_analysis method.")

        #run analysis
        self.frequency_list, self.period_list, self.amplitude_matrix, self.power_matrix, self.phase_matrix, _, fig_f,fig_p =_run_frequency_time_analysis(self.Z,self.psd,self.t,self.L,
                                     data_per_period=data_per_period,period_name=period_name,t_unit=t_unit,plot_frequency=plot_frequency,plot_period=plot_period,logplot_frequency=logplot_frequency,logplot_period=logplot_period,normalise_plots=normalise_plots,height_variable=height_variable,height_unit=height_unit)

        if fig_f is not None:
            # self.figure_frequency_time = fig_f
            self.figures.get('cissa').update({'figure_frequency_time':fig_f})
        if fig_p is not None:
            # self.figure_period_time = fig_p
            self.figures.get('cissa').update({'figure_period_time':fig_p})

        #add the results to the results dictionary
        results = self.results
        results.get('cissa').update({'frequency_time_results':{
            'frequency_list'   : self.frequency_list,
            'period_list'      : self.period_list,
            'amplitude_matrix' : self.amplitude_matrix,
            'power_matrix'     : self.power_matrix,
            'phase_matrix'     : self.phase_matrix, }
            })

        results.get('cissa').setdefault('model parameters', {})
        results.get('cissa').get('model parameters').update({
            'data_per_period'   : data_per_period,
            'period_name'       : period_name,
            't_unit'            : t_unit,
            })

        self.results = results

        #add input parameters to the class
        self.data_per_period = data_per_period
        self.period_name     = period_name
        self.t_unit          = t_unit
        if plt.get_fignums(): plt.close('all')

        return self
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def post_analyse_trend(self,
                      trend_type:        str = 'rolling_OLS',
                      t_unit:            str = '',
                      data_unit:         str = '',
                      alphas:            list = [x/20 for x in range(1,20)],
                      timestep:          float|None = None,
                      timestep_unit:     str = None,
                      include_data:      bool = True,
                      legend_loc:        int = 2,
                      shade_area:        bool = True,
                      xaxis_rotation:    float = 270,
                      window:            int = 12
                      ):
        '''
        Method to calculate and generate the trend slope and confidence for the "trend" component of the CiSSA results.
        Currently can be done using linear regression or rolling ordinary least squares.

        Parameters
        ----------
        trend_type : str, optional
            DESCRIPTION: The type of regression to perform. Current options are "linear" or "rolling_OLS". The default is 'rolling_OLS'.
        t_unit : str, optional
            DESCRIPTION: Time unit. Not required if time is a datetime. The default is ''.
        data_unit : str, optional
            DESCRIPTION. Data unit. The default is ''.
        alphas : list, optional
            DESCRIPTION. A list of significance levels for the confidence interval. For example, alpha = [.05] returns a 95% confidence interval. The default is [0.05] + [x/20 for x in range(1,20)].
        timestep : float, optional
            DESCRIPTION. Numeric timestep size in timestep_unit or seconds if the time arrayis a date. The default is None.
        timestep_unit : str, optional
            DESCRIPTION. Timestep unit (e.g. seconds, days, years). The default is None.
        include_data : bool, optional
            DESCRIPTION. Whether to include the original time-series in the plot or not. The default is True.
        legend_loc : int, optional
            DESCRIPTION: Location of the legend. The default is 2.
        shade_area : bool, optional
            DESCRIPTION: Whether to shade below the trend or not. The default is True.
        xaxis_rotation : float, optional
            DESCRIPTION: Angle (degrees) to control of the x-axis ticks. The default is 270.
        window : int, optional
            DESCRIPTION. Only used if trend_type = "rolling_OLS". Length of the rolling window. Must be strictly larger than the number of variables in the model. The default is 12.


        '''
        #check that all necessary input variables exist
        necessary_attributes = ["t","results"]
        for attr_i in necessary_attributes:
            if not hasattr(self, attr_i): raise ValueError(f"Attribute {attr_i} does not appear to exist in the class. Please fun the pycissa fit method before running the run_frequency_time_analysis method.")


        if trend_type == 'linear':
            from pycissa.postprocessing.trend.trend_functions import trend_linear

            figure_trend, self.trend_slope, self.trend_increasing_probability, self.trend_increasing_probability_text, self.trend_confidence = trend_linear(
                             self.results.get('cissa').get('components').get('trend').get('reconstructed_data'),
                             self.t,
                             t_unit=t_unit,
                             Y_unit=data_unit,
                             alphas=alphas,
                             timestep=timestep,
                             timestep_unit=timestep_unit,
                             include_data=include_data,
                             legend_loc=legend_loc,
                             shade_area=shade_area,
                             xaxis_rotation=xaxis_rotation
                             )
            self.trend_type = 'Linear'
            self.figures.get('cissa').update({'figure_trend':figure_trend})
            #
        elif trend_type == 'rolling_OLS':
            from pycissa.postprocessing.trend.trend_functions import trend_rolling
            figure_trend, self.trend_slope, self.trend_increasing_probability, self.trend_increasing_probability_text, self.trend_confidence = trend_rolling(
                              self.results.get('cissa').get('components').get('trend').get('reconstructed_data'),
                              self.t,
                              t_unit=t_unit,
                              Y_unit=data_unit,
                              window=window,
                              alphas=alphas,
                              timestep=timestep,
                              timestep_unit=timestep_unit,
                              include_data=include_data,
                              legend_loc=legend_loc,
                              shade_area=shade_area,
                              xaxis_rotation=xaxis_rotation
                              )
            self.trend_type = 'rolling_OLS'
            self.figures.get('cissa').update({'figure_trend':figure_trend})

        else:
            raise ValueError(f"Input value trend_type = {trend_type} is incorrect. Please use one of 'linear' or 'rolling_OLS'.")


       #update results dictionary
        results = self.results
        results.get('cissa').setdefault('trend results', {})
        results.get('cissa').get('trend results').setdefault(self.trend_type, {})
        results.get('cissa').get('trend results').get(self.trend_type).update({
            'trend_slope'                       : self.trend_slope,
            'trend_increasing_probability'      : self.trend_increasing_probability,
            'trend_increasing_probability_text' : self.trend_increasing_probability_text,
            'trend_confidence'                  : self.trend_confidence
            })
        self.results = results
        if plt.get_fignums(): plt.close('all')
        return self
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def post_run_monte_carlo_analysis(self,
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
                                 generate_toeplitz_matrix: bool = False):
        '''
        Function to run a monte carlo significance test on components of a signal, extracted via CiSSA.
        Signal psd/eigenvalues are compared to those obtained by applying CiSSA to surrogate data.
        Surrogates are generated using one of three available algorithms:
            random_permutation: randomly shuffle the input data
            small_shuffle: the small shuffle method of Nakamura, T., & Small, M. (2005). Small-shuffle surrogate data: Testing for dynamics in fluctuating data with trends. Physical Review E, 72(5), 056216.
            ar1_fit: Fits an autoregressive model of order 1 to the data.

        Parameters
        ----------
        alpha : float, optional
            DESCRIPTION: Significance level for surrogate hypothesis test. For example, --> 100*(1-alpha)% confidence interval. The default is 0.05 (a 95% confidence interval).
        K_surrogates : int, optional
            DESCRIPTION: Multiplier for number of surrogates. Number of surrogate data allowed to be larger than the signal and signal to still be significant = K_surrogates - 1.
                            For a one-sided test, the number of surrogate data series generated is K_surrogates/alpha - 1. For a two sided test it is 2*K_surrogates/alpha - 1.
                            The default is 1.
        surrogates : str, optional
            DESCRIPTION: The type of surrogates to generate for the hypothesis test.
                            One of "random_permutation", "small_shuffle", "ar1_fit".
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

        '''
        from pycissa.postprocessing.monte_carlo.montecarlo import run_monte_carlo_test
        #check that all necessary input variables exist
        necessary_attributes = ["psd","L","results"]
        for attr_i in necessary_attributes:
            if not hasattr(self, attr_i): raise ValueError(f"Attribute {attr_i} does not appear to exist in the class. Please run the pycissa fit method before running the post_run_monte_carlo_analysis method.")

        mc_results, figure_monte_carlo = run_monte_carlo_test(x = self.x,
                             L = self.L,
                             psd=self.psd,
                             Z=self.Z,
                             results=self.results.get('cissa'),
                             frequencies = self.frequencies,
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
                             generate_toeplitz_matrix=generate_toeplitz_matrix
                                 )
        self.results.get('cissa').update(mc_results)
        self.results['cissa']['model parameters'].update({'monte_carlo_surrogate_type':surrogates})
        self.results['cissa']['model parameters'].update({'monte_carlo_alpha':alpha})
        self.figures.get('cissa').update({'figure_monte_carlo':figure_monte_carlo})
        if plt.get_fignums(): plt.close('all')

        self.information_text += f'''
        ------------------------------------------------------
        MONTE CARLO SIGNIFICANCE TESTING
        '''
        for component_j in self.results.get('cissa').get('components'):
            if self.results.get('cissa').get('components').get(component_j).get('monte_carlo').get(surrogates).get('alpha').get(alpha).get('pass'):
                self.information_text += f'''
        Unitless frequency: {component_j} SIGNIFICANT.
                '''
        return self
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def post_group_manual(self,
                                I:                         int|float|dict,
                                season_length:             int = 1,
                                cycle_length:              list = [1.5,8],
                                include_noise:             bool = True,):
        '''
        GROUP - Manual Grouping step of CiSSA.  https://doi.org/10.1016/j.sigpro.2020.107824.

        This function groups the reconstructed components by frequency
        obtained with CiSSA into disjoint subsets and computes the share of the
        corresponding PSD.

        Syntax:     [rc, sh, kg] = group(Z,psd,I)

        Conversion from Matlab, https://github.com/jbogalo/CiSSA


        Parameters
        ----------
        I : multiple
            DESCRIPTION:
                 Four options:
                 1) A positive integer. It is the number of data per year in
                 time series. The function automatically computes the
                 trend (oscillations with period greater than 8 years), the
                 business cycle (oscillations with period between 8 & 1.5 years)
                 and seasonality.
                 2) A dictionary. Each value contains a numpy row vector with the desired
                 values of k to be included in a group, k=0,1,2,...,L/2-1. The function
                 computes the reconstructed components for these groups.
                 3) A number between 0 & 1. This number represents the accumulated
                 share of the psd achieved with the sum of the share associated to
                 the largest eigenvalues. The function computes the original
                 reconstructed time series as the sum of these components.
                 4) A number between -1 & 0. It is a percentile (in positive) of
                 the psd. The function computes the original reconstructed time
                 series as the sum of the reconstructed componentes by frequency
                 whose psd is greater that this percentile.
        season_length : int, optional
            DESCRIPTION: The default is 1. Only used for case 1. when I = A positive integer. Can be modified in the case that the season is not equal to the number of data per year. For example, if a "season" is 2 years, we enter I = 365 (for days in year) and 2 for season_length because the season is 365*2, or data_per_year*season_length.
        cycle_length : list, optional
            DESCRIPTION: The default is [1.5,8]. List of longer term cycle periods. Only used for case 1.
        include_noise : bool, optional
            DESCRIPTION: The default is True. Output noise as a vector component or not. Only used for case 1.
        '''

        from pycissa.postprocessing.grouping.grouping_functions import group
        #check that all necessary input variables exist
        necessary_attributes = ["Z","psd","L","results"]
        for attr_i in necessary_attributes:
            if not hasattr(self, attr_i): raise ValueError(f"Attribute {attr_i} does not appear to exist in the class. Please run the pycissa fit method before running the post_group_components method.")

        rc,sh,kg,psd_sh = group(self.Z,
                         self.psd,
                         I,
                         )
        self.results['cissa']['manual'] = {}
        self.results['cissa']['manual']['rc'] = rc
        self.results['cissa']['manual']['sh'] = sh
        self.results['cissa']['manual']['kg'] = kg
        self.results['cissa']['manual']['psd_sh'] = psd_sh


        return self

    def post_group_components(self,
                                 grouping_type:            str = 'monte_carlo',
                                 eigenvalue_proportion:    float = 0.9,
                                 number_of_groups_to_drop: int = 5,
                                 include_trend:            bool = True,
                                 check_noise_statistics:   bool = True,
                                 noise_alpha:              float = 0.05,
                                 ljung_box_lags:           int = 12,
                                 plot_result:              bool = True,):
        '''
        Function to group components into trend, periodic, or noise/residual.

        Parameters
        ----------
        grouping_type : str, optional
            DESCRIPTION. The default is 'monte_carlo'. Current options are 'monte_carlo', 'smallest_proportion', or 'smallest_n'.
        eigenvalue_proportion : float, optional
            DESCRIPTION. The default is 0.9. Only used of grouping type = 'smallest_proportion'
                    There are two options:
                    1) A number between 0 & 1. This number represents the accumulated
                    share of the psd achieved with the sum of the share associated to
                    the largest eigenvalues. The function computes the trend and periodic components
                    as these components, and the remaining as noise.
                    2) A number between -1 & 0. It is a percentile (in positive) of
                    the psd. The function classifies as trend/periodic the componentes by frequency
                    whose psd is greater that this percentile, and noise otherwise.
        number_of_groups_to_drop : int, optional
            DESCRIPTION. The default is 5. Only used if grouping_type == 'smallest_n'. Will order the components by psd proportion and classify the lowest "number_of_groups_to_drop" as noise.
        include_trend : bool, optional
            DESCRIPTION. The default is True. Only used if grouping_type == 'smallest_n'. If False, the trend will always be removed.
        check_noise_statistics : bool, optional
            DESCRIPTION. The default is True. If True, will check the noise component for normality and autocorrelation.
        noise_alpha : float, optional
            DESCRIPTION. The default is 0.05. Significance level for statistical tests.
        ljung_box_lags : int, optional
            DESCRIPTION. The default is 12. number of lags to check in the Ljung-box test.
        plot_result : bool, optional
            DESCRIPTION. The default is True. Plot resulting breakdown of components or not.

        '''
        def combine_components(temp_results,group_indices):
            x_grouped = np.zeros(temp_results['components']['trend']['reconstructed_data'].shape)
            for key_j in temp_results['components'].keys():
                if temp_results['components'][key_j]['array_position'] in group_indices:
                    x_grouped += temp_results['components'][key_j]['reconstructed_data']
            return x_grouped

        from pycissa.postprocessing.grouping.grouping_functions import classify_smallest_n_components
        #check that all necessary input variables exist
        necessary_attributes = ["psd","L","results"]
        for attr_i in necessary_attributes:
            if not hasattr(self, attr_i): raise ValueError(f"Attribute {attr_i} does not appear to exist in the class. Please run the pycissa fit method before running the post_group_components method.")
        # 'monte_carlo', 'smallest_proportion', or 'smallest_n'.
        if grouping_type == 'monte_carlo':
            if self.results.get('cissa').get('components').get('trend').get('monte_carlo') is None: raise ValueError(f"Please run the post_run_monte_carlo_analysis method before running the post_group_components with grouping_type == 'monte_carlo' or use another grouping type.")
            from pycissa.postprocessing.grouping.grouping_functions import classify_monte_carlo_non_significant_components
            trend,  periodic, noise  = classify_monte_carlo_non_significant_components(self.Z,
                                                                                      self.results.get('cissa'))
        elif grouping_type == 'smallest_proportion':
            from pycissa.postprocessing.grouping.grouping_functions import classify_smallest_proportion_psd
            trend,  periodic, noise = classify_smallest_proportion_psd(self.Z,
                                                                       self.psd,
                                                                       self.L,
                                                                       eigenvalue_proportion)
        elif grouping_type == 'smallest_n':
            from pycissa.postprocessing.grouping.grouping_functions import classify_smallest_n_components
            trend,  periodic, noise = classify_smallest_n_components(self.Z,
                                                                     self.psd,
                                                                     self.L,
                                                                     number_of_groups_to_drop,
                                                                     include_trend=include_trend)
        else: raise ValueError(f"Input parameter 'grouping_type' should be one of 'monte_carlo', 'smallest_proportion', or 'smallest_n'. You entered: {grouping_type}.")

        #get share of psd for each component
        trend_share,periodic_share,noise_share = 0.,0.,0.
        for key_j in self.results['cissa']['components'].keys():
            index = self.results['cissa']['components'][key_j]['array_position']
            share = self.results['cissa']['components'][key_j]['percentage_share_of_psd']
            if index in trend:    trend_share += share
            if index in periodic: periodic_share += share
            if index in noise:    noise_share += share



        self.results['cissa']['noise component tests'].update({'trend_index':trend})
        self.results['cissa']['noise component tests'].update({'trend_share':trend_share})
        self.results['cissa']['noise component tests'].update({'periodic_index':periodic})
        self.results['cissa']['noise component tests'].update({'periodic_share':periodic_share})
        self.results['cissa']['noise component tests'].update({'noise_index':noise})
        self.results['cissa']['noise component tests'].update({'noise_share':noise_share})

        self.x_trend = combine_components(self.results['cissa'],trend)
        self.x_periodic = combine_components(self.results['cissa'],periodic)
        self.x_noise = combine_components(self.results['cissa'],noise)


        if check_noise_statistics:
            from pycissa.postprocessing.statistics.von_neumann import rank_von_neumann_test
            from pycissa.postprocessing.statistics.ljung_box import run_ljung_box_test
            from pycissa.postprocessing.statistics.normality_tests import run_normality_test
            _,_,_,interpretation = rank_von_neumann_test(self.x_noise,alpha = noise_alpha)
            self.results['cissa']['noise component tests'].update({'rank von Neumann' : interpretation})
            _,_,_,_,interpretation = run_ljung_box_test(self.x_noise,lags = ljung_box_lags,alpha = noise_alpha)
            self.results['cissa']['noise component tests'].update({'ljung_box' : interpretation})
            _,_,_,interpretation = run_normality_test(self.x_noise,alpha = noise_alpha)
            self.results['cissa']['noise component tests'].update({'normality' : interpretation})


        if plot_result:
            from pycissa.utilities.plotting import plot_grouped_components,plot_noise_residual
            fig = plot_grouped_components(self.t,
                                          self.x,
                                          self.x_trend,
                                          self.x_periodic,
                                          self.x_noise,)
            self.figures.get('cissa').update({'figure_split_components':fig})

            fig = plot_noise_residual(self.x, self.x_noise)
            self.figures.get('cissa').update({'figure_residual_check':fig})



        if plt.get_fignums(): plt.close('all')

        self.information_text += f'''
        ------------------------------------------------------
        COMPONENT VARIANCE
        TREND   : {self.results.get('cissa').get('noise component tests').get('trend_share')}%
        PERIODIC: {self.results.get('cissa').get('noise component tests').get('periodic_share')}%
        NOISE   : {self.results.get('cissa').get('noise component tests').get('noise_share')}%
        '''

        return self

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def post_periodogram_analysis(self,
                                  significant_components             : list|None = None,
                                  monte_carlo_significant_components : bool = True,
                                  alpha                              : float = 0.05,
                                  max_breakpoints                    : int = 1,
                                  n_boot                             : int = 500,
                                  hurst_window                       : int = 12,
                                  normalization                      : str = 'standard',
                                  center_data                        : bool = True,
                                  fit_mean                           : bool = True,
                                  nterms                             : int = 1,
                                  **kwargs):
        '''
        Function to run a periodogram analysis to find the fractal scaling of the time series.
        In all cases the trend is not considered, and significant periodic components can be ignored too using the input parameters.
        Also calculates the Hurst exponent for the full and detrended series.

        Parameters
        ----------
        significant_components : list|None, optional
            DESCRIPTION. The default is None. A list of significant components which will not be considered in the periodogram analysis. Can also be None, in which case all components (except the trend) will be used for the periodogram, or if significant_components = None and monte_carlo_significant_components = True, then the monte carlo significant components will be removed.
        monte_carlo_significant_components : bool, optional
            DESCRIPTION. The default is True. If significant_components = None and monte_carlo_significant_components = True, the significant components list will be filled with the significant components as defined using the monte carlo analysis.
        alpha : float, optional
            DESCRIPTION. The default is 0.05. Significance level for statistical tests.
        max_breakpoints : int, optional
            DESCRIPTION. The default is 1. Max number of breakpoints for the segmented linear fit. Currently will always be reset to 1 if >1.
        n_boot : int, optional
            DESCRIPTION. The default is 500. Number of bootstrap iterations for the segmented linear fit.
        hurst_window : int, optional
            DESCRIPTION. The default is 12. The window length (in number of time steps) for the rolling Hurst calculation.
        normalization : str, optional
            DESCRIPTION. The default is 'standard'. How to normalise the lomb-scargle periodogram.
        center_data :  bool, optional
            DESCRIPTION. The default is True. Whether to center the data for the lomb-scargle periodogram.
        fit_mean :  bool, optional
            DESCRIPTION. The default is True. Whether to fit the mean for the lomb-scargle periodogram.
        nterms : int, optional
            DESCRIPTION. The default is 1. Number of terms in the trigonometric fit of the lomb-scargle periodogram.
        **kwargs
            DESCRIPTION. keyword arguments for fitting.

        '''
        from pycissa.postprocessing.periodogram.periodogram import generate_peridogram_plots,generate_lomb_scargle_peridogram_plots
        necessary_attributes = ["psd","frequencies","results"]
        for attr_i in necessary_attributes:
            if not hasattr(self, attr_i): raise ValueError(f"Attribute {attr_i} does not appear to exist in the class. Please run the pycissa fit method before running the post_periodogram_analysis method.")
        necessary_attributes = ["x_trend","x_periodic",'x_noise']
        for attr_i in necessary_attributes:
            if not hasattr(self, attr_i): raise ValueError(f"Attribute {attr_i} does not appear to exist in the class. Please run the pycissa auto_detrend method before running the post_group_components method.")

        #if no significant components supplied (i.e. = None) and if choosing to use monte_carlo to define significant components, here get those significant components.
        if (not significant_components) &  (monte_carlo_significant_components):
            if self.results.get('cissa').get('components').get('trend').get('monte_carlo') is None: raise ValueError(f"Please run the post_run_monte_carlo_analysis method before running the post_periodogram_analysis with monte_carlo_significant_components = True and no explicitly set significant_components.")
            mc_surrogates = self.results['cissa']['model parameters']['monte_carlo_surrogate_type']
            mc_alpha = self.results['cissa']['model parameters']['monte_carlo_alpha']
            significant_components = []
            for key_j in self.results['cissa']['components'].keys():
                if key_j != 'trend':
                    if self.results['cissa']['components'][key_j]['monte_carlo'][mc_surrogates]['alpha'][mc_alpha]['pass']:
                        position = self.results['cissa']['components'][key_j]['array_position']
                        significant_components.append(position)
            if len(significant_components) == 0:
                significant_components = None

        fig_linear, fig_segmented, fig_robust_linear, linear_slopes, segmented_slopes, robust_linear_slopes,all_hurst,detrended_hurst,fig_rolling_hurst,rolling_hurst,rolling_hurst_detrended,fig_robust_segmented,robust_segmented_results  =generate_peridogram_plots(self.x_trend,self.x_periodic+self.x_noise,self.psd,self.frequencies,significant_components=significant_components,alpha=alpha,max_breakpoints=max_breakpoints,n_boot=n_boot,hurst_window=hurst_window)
        self.figures.get('cissa').update({'figure_periodogram_linear'           :fig_linear})
        self.figures.get('cissa').update({'figure_periodogram_robust_linear'    :fig_robust_linear})
        self.figures.get('cissa').update({'figure_periodogram_segmented'        :fig_segmented})
        self.figures.get('cissa').update({'figure_periodogram_robust_segmented' :fig_robust_segmented})
        self.figures.get('cissa').update({'figure_rolling_Hurst'                :fig_rolling_hurst})



        self.results.get('cissa').get('fractal scaling results').update({'linear_periodogram_slopes'           : linear_slopes,
                                                                         'robust_linear_periodogram_slopes'    : robust_linear_slopes,
                                                                         'segmented_periodogram_slopes'        : segmented_slopes,
                                                                         'full Hurst exponent'                 : all_hurst,
                                                                         'detrended Hurst exponent'            : detrended_hurst,
                                                                         'rolling Hurst exponent'              : rolling_hurst,
                                                                         'detrended rolling Hurst exponent'    : rolling_hurst_detrended,
                                                                         'robust_segmented_periodogram_slopes' : robust_segmented_results})



        if plt.get_fignums(): plt.close('all')
        return self
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def plot_autocorrelation(self,
                                noise_components: list|None = None,
                                monte_carlo_noise:bool = False,
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
                                title_size:       float|int=14,
                                label_size:       float|int=12
                                ):
        '''
        Function to generate and return a figure with three subplots:
            1) time series data,
            2) the autocorrelation function of the data,
            3) the partial autocorrelation function of the data.
        Option to create a second plot with noise components after running cissa.

        Parameters
        ----------
        self.t : np.ndarray
            DESCRIPTION: array of input times/dates.
        self.x : np.ndarray
            DESCRIPTION: array of input data.
        noise_components : list|None, optional
            DESCRIPTION. The default is None. A list of noise components. If provided a second figure will be created with the autocorrelation of the sum of all noise components in the noise_components list.
        monte_carlo_noise : bool, optional
            DESCRIPTION. The default is False. If True, the noise_components are ignored and the components that fail the monte carlo analysis are combined and used as noise.
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
        title_size : float|int, optional
            DESCRIPTION. The default is 14.
        label_size : float|int, optional
            DESCRIPTION. The default is 12.

        Returns
        -------
        fig : matplotlib figure
            DESCRIPTION. Figure of time series, acf, pacf.

        '''
        #----------------------------------------------------------------------
        #ensure data is not uncensored or nan
        if self.censored:  raise ValueError("Censored data detected. Please run pre_fix_censored_data before fitting.")
        if self.isnan: raise ValueError("WARNING: nan data detected. Please run pre_fill_gaps before fitting.")
        #----------------------------------------------------------------------
        from pycissa.postprocessing.statistics.autocorrelation_function import plot_time_series_and_acf_pacf

        fig = plot_time_series_and_acf_pacf(self.t,self.x,
                                          acf_lags=acf_lags,
                                          pacf_lags=pacf_lags,
                                          alpha=alpha,
                                          use_vlines=use_vlines,
                                          adjusted=adjusted,
                                          fft=fft,
                                          missing=missing,
                                          zero=zero,
                                          auto_ylims=auto_ylims,
                                          bartlett_confint=bartlett_confint,
                                          pacf_method=pacf_method,
                                          acf_color=acf_color,
                                          pacf_color=pacf_color,
                                          title = 'Original Time Series',
                                          title_size=title_size,
                                          label_size=label_size
                                          )
        self.figures.get('cissa').update({'figure_autocorrelation':fig})


        ######################################################################
        if monte_carlo_noise:
            #check that fit has been run
            necessary_attributes = ["results"]
            for attr_i in necessary_attributes:
                if not hasattr(self, attr_i): raise ValueError(f"Attribute {attr_i} does not appear to exist in the class. Please fun the pycissa fit method before running the plot_autocorrelation method with noise components.")

            #check that monte carlo has been run
            monte_carlo_complete = self.results['cissa']['components']['trend'].get('monte_carlo',False)
            if not monte_carlo_complete:raise ValueError(f"Please run the post_run_monte_carlo_analysis method before running the plot_autocorrelation method with monte_carlo_noise = True.")


            monte_carlo_type = self.results['cissa']['model parameters']['monte_carlo_surrogate_type']
            monte_carlo_alpha = self.results['cissa']['model parameters']['monte_carlo_alpha']
            noise_components = []
            for key_j in self.results['cissa']['components'].keys():
                if not self.results['cissa']['components'][key_j]['monte_carlo'][monte_carlo_type]['alpha'][monte_carlo_alpha]['pass']:
                    noise_components.append(self.results['cissa']['components'][key_j]['array_position'])
        ######################################################################



        ######################################################################
        if noise_components is not None:
            #check that fit has been run
            necessary_attributes = ["results"]
            for attr_i in necessary_attributes:
                if not hasattr(self, attr_i): raise ValueError(f"Attribute {attr_i} does not appear to exist in the class. Please fun the pycissa fit method before running the plot_autocorrelation method with noise components.")


            noise_array = np.zeros(self.results['cissa']['components']['trend']['reconstructed_data'].shape)
            for key_j in self.results['cissa']['components'].keys():
                if self.results['cissa']['components'][key_j]['array_position'] in noise_components:
                    noise_array += self.results['cissa']['components'][key_j]['reconstructed_data']


            fig2 = plot_time_series_and_acf_pacf(self.t,noise_array,
                                              acf_lags=acf_lags,
                                              pacf_lags=pacf_lags,
                                              alpha=alpha,
                                              use_vlines=use_vlines,
                                              adjusted=adjusted,
                                              fft=fft,
                                              missing=missing,
                                              zero=zero,
                                              auto_ylims=auto_ylims,
                                              bartlett_confint=bartlett_confint,
                                              pacf_method=pacf_method,
                                              acf_color=acf_color,
                                              pacf_color=pacf_color,
                                              title = 'Residual Time Series',
                                              title_size=title_size,
                                              label_size=label_size
                                              )
            self.figures.get('cissa').update({'figure_autocorrelation_noise':fig2})
        ######################################################################
        if plt.get_fignums(): plt.close('all')
        return self



    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def plot_seasonal_boxplots(self,
                               split_date:    datetime|None = None,
                               bar_width:     float = 0.25,
                               plot_type:     str = 'both',
                               remove_trend:  bool=False,
                               include_noise: bool=True,):
        '''
        Function to plot seasonal (either monthly or yearly or both) boxplots.
        Includes the option to split the data by a set date and will plot grouped boxplots with one group being the boxplot of the seasonal data before the set date, the other after.

        Parameters
        ----------
        split_date : datetime|None, optional
            DESCRIPTION. The default is None. A datetime object which splits the boxplots into groups, one before the set date, one after. If None, the data is not split.
        bar_width : float, optional
            DESCRIPTION. The default is 0.25. Width of each individual bar
        plot_type : str, optional
            DESCRIPTION. The default is 'both'. One of 'both', 'monthly', or 'yearly', depending on the type of boxplot to be plotted.
        remove_trend : bool, optional
            DESCRIPTION. The default is False. If True, the trend is remove. If True, then we first must have run post_group_components.
        include_noise : bool, optional
            DESCRIPTION. The default is True. Only used if remove_trend = True. If False, the noise component is removed, leaving only the periodic component to be box-plotted.


        '''
        #----------------------------------------------------------------------
        #ensure data is not uncensored or nan
        if self.censored:  raise ValueError("Censored data detected. Please run pre_fix_censored_data before plot_seasonal_boxplots.")
        if self.isnan: raise ValueError("WARNING: nan data detected. Please run pre_fill_gaps before plot_seasonal_boxplots.")
        #----------------------------------------------------------------------

        if plot_type in ['both','monthly']:
            from pycissa.utilities.plotting import seasonal_boxplots
            if remove_trend:
                necessary_attributes = ["x_trend","x_periodic","x_noise"]
                for attr_i in necessary_attributes:
                    if not hasattr(self, attr_i): raise ValueError(f"Attribute {attr_i} does not appear to exist in the class. Please run the pycissa post_group_components method before running the plot_seasonal_boxplots method with remove_trend = True.")
                if include_noise:x_plot = self.x_periodic + self.x_noise
                else:x_plot = self.x_periodic
            else:
                x_plot = self.x
            fig = seasonal_boxplots(self.t,x_plot,split_date=split_date,bar_width=bar_width)
            self.figures.get('cissa').update({'figure_monthly_seasonal_box':fig})

        if plot_type in ['both','yearly']:
            from pycissa.utilities.plotting import yearly_boxplots
            if remove_trend:
                necessary_attributes = ["x_trend","x_periodic","x_noise"]
                for attr_i in necessary_attributes:
                    if not hasattr(self, attr_i): raise ValueError(f"Attribute {attr_i} does not appear to exist in the class. Please run the pycissa post_group_components method before running the plot_seasonal_boxplots method with remove_trend = True.")
                if include_noise:x_plot = self.x_periodic + self.x_noise
                else:x_plot = self.x_periodic
            else:
                x_plot = self.x
            fig = yearly_boxplots(self.t,x_plot,bar_width=bar_width)
            self.figures.get('cissa').update({'figure_yearly_seasonal_box':fig})
        if plt.get_fignums(): plt.close('all')
        return self
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def auto_denoise(self,
                     L:             int = None,
                     plot_denoised: bool = True,
                     **kwargs):
        '''
        Function to automatically denoise a time series using Cissa.
        Automatically:
            corrects censored and nan values.
            fits the time series using Cissa.
            groups the components into signal and noise.
            Plots the results.

        Parameters
        ----------
        L : int, optional
            DESCRIPTION. The default is None. The default is None. CiSSA window length.
        plot_denoised : bool, optional
            DESCRIPTION. The default is True.  If True, the resulting denoised figure is plotted.
        **kwargs : TYPE
            DESCRIPTION. key word arguments for the auto_fix_censoring_nan(), fit(), post_run_monte_carlo_analysis(), and  post_group_components() functions.

        '''
        #if L is not provided then we take L as (the floor of) half the series length
        if not L:
            L = int(np.floor(len(self.x)/2))

        #fix censoring and nan
        _ = self.auto_fix_censoring_nan(L,**kwargs)

        #run cissa
        _ = self.fit(
                L,
                extension_type = kwargs.get('extension_type','AR_LR'),
                multi_thread_run = kwargs.get('multi_thread_run',True),
                generate_toeplitz_matrix = kwargs.get('generate_toeplitz_matrix',False))

        #run monte carlo if needed
        if kwargs.get('grouping_type','monte_carlo')=='monte_carlo':
            _ = self.post_run_monte_carlo_analysis(
                                         alpha                    = kwargs.get('alpha',0.05),
                                         K_surrogates             = kwargs.get('K_surrogates',1),
                                         surrogates               = kwargs.get('surrogates','random_permutation'),
                                         seed                     = kwargs.get('seed',None),
                                         sided_test               = kwargs.get('sided_test','one sided'),
                                         remove_trend             = kwargs.get('remove_trend',True),
                                         trend_always_significant = kwargs.get('trend_always_significant',True),
                                         A_small_shuffle          = kwargs.get('A_small_shuffle',1.),
                                         extension_type           = kwargs.get('extension_type','AR_LR'),
                                         multi_thread_run         = kwargs.get('multi_thread_run',False),
                                         generate_toeplitz_matrix = kwargs.get('generate_toeplitz_matrix',False),
                                         )

         #   run grouping
        _ = self.post_group_components(
                                      grouping_type            = kwargs.get('grouping_type','monte_carlo'),
                                      eigenvalue_proportion    = kwargs.get('eigenvalue_proportion',0.9),
                                      number_of_groups_to_drop = kwargs.get('number_of_groups_to_drop',5),
                                      include_trend            = kwargs.get('include_trend',True),
                                      check_noise_statistics   = kwargs.get('check_noise_statistics',True),
                                      noise_alpha              = kwargs.get('noise_alpha',0.05),
                                      ljung_box_lags           = kwargs.get('ljung_box_lags',12),
                                      plot_result              = kwargs.get('plot_result',True))

        self.x_denoised = self.x_trend + self.x_periodic

        if plot_denoised:
            from pycissa.utilities.plotting import plot_denoised_signal
            fig = plot_denoised_signal(self.t,
                                 self.x,
                                 self.x_denoised
                                        )
            self.figures.get('cissa').update({'figure_denoised':fig})
        if plt.get_fignums(): plt.close('all')

        return self

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def auto_detrend(self,
                     L:           int = None,
                     plot_result: bool = True,
                     **kwargs):
        '''
        Function to automatically detrend a signal using Cissa.
        Automatically:
            corrects censored and nan values.
            fits the time series using Cissa.
            groups the components into trend and detrended signal.
            Plots the results.

        Parameters
        ----------
        L : int, optional
            DESCRIPTION. The default is None. CiSSA window length.
        plot_result : bool, optional
            DESCRIPTION. The default is True. If True, the resulting detrended figure is plotted.
        **kwargs : dict
            DESCRIPTION. key word arguments for the auto_fix_censoring_nan() and fit() functions.
        '''
        #if L is not provided then we take L as (the floor of) half the series length
        if not L:
            L = int(np.floor(len(self.x)/2))

        #fix censoring and nan
        _ = self.auto_fix_censoring_nan(L,**kwargs)

        #run cissa
        _ = self.fit(
                L,
                extension_type = kwargs.get('extension_type','AR_LR'),
                multi_thread_run = kwargs.get('multi_thread_run',True),
                generate_toeplitz_matrix = kwargs.get('generate_toeplitz_matrix',False))

        #group components
        from pycissa.postprocessing.grouping.grouping_functions import group
        number_of_signal_components = int(len(self.psd)/2 - 1)
        I = {'trend'    :[0],
             'detrended':[x for x in range(1,number_of_signal_components+1)]}
        rc, sh, kg, psd_sh = group(self.Z,self.psd,I)

        self.x_trend = rc['trend'].reshape(len(rc['trend']),)
        self.x_detrended = rc['detrended'].reshape(len(rc['detrended']),)


        if plot_result:
            from pycissa.utilities.plotting import plot_detrended_signal
            fig=plot_detrended_signal(self.t,
                                    self.x,
                                    self.x_trend,
                                    self.x_detrended,
                                        )

            self.figures.get('cissa').update({'figure_detrended':fig})

        #analyse the trend
        _ = self.post_analyse_trend(
                          trend_type     = kwargs.get('trend_type','rolling_OLS'),
                          t_unit         = kwargs.get('t_unit',''),
                          data_unit      = kwargs.get('data_unit',''),
                          alphas         = kwargs.get('alphas',[x/20 for x in range(1,20)]),
                          timestep       = kwargs.get('timestep',None),
                          timestep_unit  = kwargs.get('timestep_unit',None),
                          include_data   = kwargs.get('include_data',True),
                          legend_loc     = kwargs.get('legend_loc',2),
                          shade_area     = kwargs.get('shade_area',True),
                          xaxis_rotation = kwargs.get('xaxis_rotation',270),
                          window         = kwargs.get('window',12)
                          )

        if plt.get_fignums(): plt.close('all')
        return self



    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def auto_fix_censoring_nan(self,L : int,**kwargs):
        '''
        Function to automatically fix any censoring or nan values in the data.

        Parameters
        ----------
        L : int
            DESCRIPTION: CiSSA window length.
        **kwargs : dict
            DESCRIPTION. key word arguments for the pre_fix_censored_data() and pre_fill_gaps() functions.


        '''
        #check for censored, nan data
        if self.censored:
            warnings.warn("Censored data detected. Running pre_fix_censored_data to fix...")
            _ = self.pre_fix_censored_data(
                                     replace_type        = kwargs.get('replace_type','raw'),
                                     lower_multiplier    = kwargs.get('lower_multiplier',0.5),
                                     upper_multiplier    = kwargs.get('upper_multiplier',1.1),
                                     default_value_lower = kwargs.get('default_value_lower',0.),
                                     default_value_upper = kwargs.get('default_value_upper',0.),
                                     hicensor_lower      = kwargs.get('hicensor_lower',False),
                                     hicensor_upper      = kwargs.get('hicensor_upper',False),
                                     )

        if self.isnan:
            warnings.warn("Censored data detected. Running pre_fill_gaps to fix...")
            from pycissa.utilities.helper_functions import get_keyword_args
            keys_to_remove = get_keyword_args(self.pre_fill_gaps)
            temp_kwargs = {key: value for key, value in kwargs.items() if key not in keys_to_remove}
            convergence_ = ['value', 0.01 * np.nanmin(self.x)]
            _ = self.pre_fill_gaps(
                          L,
                          convergence                = kwargs.get('convergence',convergence_),
                          extension_type             = kwargs.get('extension_type','AR_LR'),
                          multi_thread_run           = kwargs.get('multi_thread_run',True),
                          initial_guess              = kwargs.get('initial_guess',['previous', 1]),
                          outliers                   = kwargs.get('outliers',['nan_only',None]),
                          estimate_error             = kwargs.get('estimate_error',True),
                          test_number                = kwargs.get('test_number',10),
                          test_repeats               = kwargs.get('test_repeats',1),
                          z_value                    = kwargs.get('z_value',1.96),
                          component_selection_method = kwargs.get('component_selection_method','monte_carlo_significant_components'),
                          eigenvalue_proportion      = kwargs.get('eigenvalue_proportion',0.95),
                          number_of_groups_to_drop   = kwargs.get('number_of_groups_to_drop',1),
                          data_per_unit_period       = kwargs.get('data_per_unit_period',1),
                          use_cissa_overlap          = kwargs.get('use_cissa_overlap',False),
                          drop_points_from           = kwargs.get('drop_points_from','Left'),
                          max_iter                   = kwargs.get('max_iter',50),
                          verbose                    = kwargs.get('verbose',False),
                          alpha                      = kwargs.get('alpha', 0.05),
                          **temp_kwargs,
                          )

        return self

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def auto_cissa(self,
                   L: int = None,
                   **kwargs):
        '''
        AUTO-CISSA!

        Parameters
        ----------
        L : int, optional
            DESCRIPTION. The default is None.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        #if L is not provided then we take L as (the floor of) half the series length
        if not L:
            L = int(np.floor(len(self.x)/2))
            print(f"No input parameter L provided. Taking L as {L}.")

        #fix censoring and nan
        print('Checking for censored or nan data...')
        _ = self.auto_fix_censoring_nan(L,**kwargs)

        #plot original time series
        _ = self.plot_original_time_series()

        #run cissa
        print('RUNNING CISSA!')
        _ = self.fit(
                L,
                extension_type = kwargs.get('extension_type','AR_LR'),
                multi_thread_run = kwargs.get('multi_thread_run',True),
                generate_toeplitz_matrix = kwargs.get('generate_toeplitz_matrix',False))

        print('Performing monte-carlo significance analysis...')
        #run monte carlo
        _ = self.post_run_monte_carlo_analysis(
                                     alpha                    = kwargs.get('alpha',0.05),
                                     K_surrogates             = kwargs.get('K_surrogates',1),
                                     surrogates               = kwargs.get('surrogates','random_permutation'),
                                     seed                     = kwargs.get('seed',None),
                                     sided_test               = kwargs.get('sided_test','one sided'),
                                     remove_trend             = kwargs.get('remove_trend',True),
                                     trend_always_significant = kwargs.get('trend_always_significant',True),
                                     A_small_shuffle          = kwargs.get('A_small_shuffle',1.),
                                     extension_type           = kwargs.get('extension_type','AR_LR'),
                                     multi_thread_run         = kwargs.get('multi_thread_run',False),
                                     generate_toeplitz_matrix = kwargs.get('generate_toeplitz_matrix',False),
                                     )

        #run grouping
        print('Grouping components...')
        _ = self.post_group_components(
                                      grouping_type            = kwargs.get('grouping_type','monte_carlo'),
                                      eigenvalue_proportion    = kwargs.get('eigenvalue_proportion',0.9),
                                      number_of_groups_to_drop = kwargs.get('number_of_groups_to_drop',5),
                                      include_trend            = kwargs.get('include_trend',True),
                                      check_noise_statistics   = kwargs.get('check_noise_statistics',True),
                                      noise_alpha              = kwargs.get('noise_alpha',0.05),
                                      ljung_box_lags           = kwargs.get('ljung_box_lags',12),
                                      plot_result              = kwargs.get('plot_result',True))

        #plot frequency time graphs
        print('Running frequency time analysis...')
        _ = self.post_run_frequency_time_analysis(
                                        data_per_period   = kwargs.get('data_per_period',1),
                                        period_name       = kwargs.get('period_name',''),
                                        t_unit            = kwargs.get('t_unit',''),
                                        plot_frequency    = kwargs.get('plot_frequency',True),
                                        plot_period       = kwargs.get('plot_period',True),
                                        logplot_frequency = kwargs.get('logplot_frequency',True),
                                        logplot_period    = kwargs.get('logplot_period',False),
                                        normalise_plots   = kwargs.get('normalise_plots',False),
                                        height_variable   = kwargs.get('height_variable','power'),
                                        height_unit       = kwargs.get('height_unit',''))

        #plot trend
        print('Analysing trend...')
        _ = self.post_analyse_trend(
                          trend_type     = kwargs.get('trend_type','rolling_OLS'),
                          t_unit         = kwargs.get('t_unit',''),
                          data_unit      = kwargs.get('data_unit',''),
                          alphas         = kwargs.get('alphas',[x/20 for x in range(1,20)]),
                          timestep       = kwargs.get('timestep',None),
                          timestep_unit  = kwargs.get('timestep_unit',None),
                          include_data   = kwargs.get('include_data',True),
                          legend_loc     = kwargs.get('legend_loc',2),
                          shade_area     = kwargs.get('shade_area',True),
                          xaxis_rotation = kwargs.get('xaxis_rotation',270),
                          window         = kwargs.get('window',12)
                          )

        #plot autocorrelation
        print('Calculating time-series autocorrelation...')
        if not kwargs.get('noise_components'):
            kwargs.update({'monte_carlo_noise':True})
        if kwargs.get('run_autocorrelation',True):
            _ = self.plot_autocorrelation(
                                        noise_components  = kwargs.get('noise_components',None),
                                        monte_carlo_noise = kwargs.get('monte_carlo_noise',False),
                                        acf_lags          = kwargs.get('acf_lags',None),
                                        pacf_lags         = kwargs.get('pacf_lags',None),
                                        alpha             = kwargs.get('alpha',0.05),
                                        use_vlines        = kwargs.get('use_vlines',True),
                                        adjusted          = kwargs.get('adjusted',False),
                                        fft               = kwargs.get('fft',False),
                                        missing           = kwargs.get('missing','none'),
                                        zero              = kwargs.get('zero',True),
                                        auto_ylims        = kwargs.get('auto_ylims',False),
                                        bartlett_confint  = kwargs.get('bartlett_confint',True),
                                        pacf_method       = kwargs.get('pacf_method','ywm'),
                                        acf_color         = kwargs.get('acf_color','blue'),
                                        pacf_color        = kwargs.get('pacf_color','blue'),
                                        title_size        = kwargs.get('title_size',14),
                                        label_size        = kwargs.get('label_size',12)
                                        )

        # run periodogram analysis
        if kwargs.get('run_periodogram',True):
            print("running peridogram analysis")
            _ = self.post_periodogram_analysis(
                                          significant_components             = kwargs.get('significant_components',None),
                                          monte_carlo_significant_components = kwargs.get('monte_carlo_significant_components',True),
                                          alpha                              = kwargs.get('alpha',0.05),
                                          max_breakpoints                    = kwargs.get('max_breakpoints',1),
                                          n_boot                             = kwargs.get('n_boot',500),
                                          hurst_window                       = kwargs.get('hurst_window',12),
                                          )
        print("Auto Cissa Complete!")
        return self

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def auto_cissa_classic(self,
                   I:                         int|float|dict,
                   L: int = None,
                   season_length:             int = 1,
                   cycle_length:              list = [1.5,8],
                   **kwargs):
        '''
        This version of auto_cissa (classic) implements a version which is more faithful to the original Matlab version of Cissa by https://github.com/jbogalo/CiSSA.

        Parameters
        ----------
        I : multiple
            DESCRIPTION:
                 Four options:
                 1) A positive integer. It is the number of data per year in
                 time series. The function automatically computes the
                 trend (oscillations with period greater than 8 years), the
                 business cycle (oscillations with period between 8 & 1.5 years)
                 and seasonality.
                 2) A dictionary. Each value contains a numpy row vector with the desired
                 values of k to be included in a group, k=0,1,2,...,L/2-1. The function
                 computes the reconstructed components for these groups.
                 3) A number between 0 & 1. This number represents the accumulated
                 share of the psd achieved with the sum of the share associated to
                 the largest eigenvalues. The function computes the original
                 reconstructed time series as the sum of these components.
                 4) A number between -1 & 0. It is a percentile (in positive) of
                 the psd. The function computes the original reconstructed time
                 series as the sum of the reconstructed componentes by frequency
                 whose psd is greater that this percentile.
        season_length : int, optional
            DESCRIPTION: The default is 1. Only used for case 1. when I = A positive integer. Can be modified in the case that the season is not equal to the number of data per year. For example, if a "season" is 2 years, we enter I = 365 (for days in year) and 2 for season_length because the season is 365*2, or data_per_year*season_length.
        cycle_length : list, optional
            DESCRIPTION: The default is [1.5,8]. List of longer term cycle periods. Only used for case 1.
        L : int, optional
            DESCRIPTION. The default is None.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        #if L is not provided then we take L as (the floor of) half the series length
        if not L:
            L = int(np.floor(len(self.x)/2))
            print(f"No input parameter L provided. Taking L as {L}.")

        #fix censoring and nan
        print('Checking for censored or nan data...')
        _ = self.auto_fix_censoring_nan(L,**kwargs)

        #plot original time series
        _ = self.plot_original_time_series()

        #fit cissa
        self.fit(L=L,
                         extension_type   = kwargs.get('extension_type','AR_LR'),
                         multi_thread_run = kwargs.get('multi_thread_run',True),
                         num_workers      = kwargs.get('num_workers',2),
                         generate_toeplitz_matrix = kwargs.get('generate_toeplitz_matrix',False),
                         )

        #group components
        self.post_group_manual(I=I,
                               season_length  = kwargs.get('season_length',1),
                               cycle_length   = kwargs.get('cycle_length',[1.5,8]),
                               include_noise  = kwargs.get('include_noise',True),
                               )

        #reconstructed components
        rc = self.results['cissa']['manual']['rc']
        if type(I) == int or type(I) == float:
            if ((I-np.floor(I))==0) & (I>0):
                self.x_trend = rc['trend']
                self.x_seasonality = rc['seasonality']
                self.x_long_term_cycle = rc['long term cycle']
                self.x_noise = rc['noise']
        else:
            self.x_reconstructed = rc



    #List of stuff to add in here
    '''
    Fix trend.
    TESTING ALL FUNCTIONS!!!!!
    add print summary text
    function commenting!
    predict method (TO DO, maybe using MAPIE? sktime? autots?)
    add option to center data (HARD)
    add Lomb Scargle

    '''
