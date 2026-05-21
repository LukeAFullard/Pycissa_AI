import numpy as np
import warnings
import matplotlib.pyplot as plt

def initial_mdata_checks(t: np.ndarray, x: np.ndarray, use_32_bit: bool):
    """
    Data checks to ensure t,x are numpy arrays of the correct shape.
    Will try to convert to the correct shape if they are not
    x should be (T, M)
    """
    if not isinstance(x, np.ndarray):
        try:
            x = np.array(x)
        except Exception:
            raise ValueError('Input "x" is not a numpy array, nor can be converted to one.')

    myshape = x.shape
    if len(myshape) != 2:
        raise ValueError(f'Input "x" should be a 2D matrix (T, M). The size of x is ({myshape})')

    if use_32_bit:
        new_x = np.empty_like(x, dtype=object)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                try:
                    new_x[i, j] = np.float32(x[i, j])
                except (ValueError, OverflowError):
                    new_x[i, j] = x[i, j]
        x = new_x
        try:
            x = x.astype(np.float32)
        except ValueError:
            pass

    if not isinstance(t, np.ndarray):
        try:
            t = np.array(t)
            t = t.reshape(len(t),)
        except Exception:
            raise ValueError('Input "t" is not a numpy array, nor can be converted to one.')

    if len(t.shape) != 1:
        try:
            t = t.reshape(len(t),)
        except Exception:
            raise ValueError(f'Input "t" should be a 1D vector. The size of t is ({t.shape})')

    return t, x

class MCissa:
    """
    Multivariate Circulant Singular Spectrum Analysis
    """
    def __init__(self, t: np.ndarray, x: np.ndarray, use_32_bit: bool = False):
        self.use_32_bit = use_32_bit
        t, x = initial_mdata_checks(t, x, self.use_32_bit)

        self.x_raw = x
        self.t_raw = t

        #perform check for censored data
        from pycissa.preprocessing.data_cleaning.data_cleaning import detect_censored_data
        self.censored,num_censored = detect_censored_data(x.flatten())

        self.information_text = ''
        if self.censored:
            warnings.warn("WARNING: Censored data detected. Please run pre_fix_censored_data before fitting.")
            self.information_text += f'''
            ------------------------------------------------------
            {num_censored} censored data points found.
            '''

        #perform check for nan data
        from pycissa.preprocessing.data_cleaning.data_cleaning import detect_nan_data
        self.isnan = detect_nan_data(x.flatten())
        if self.isnan: warnings.warn("WARNING: nan data detected. Please run pre_fill_gaps before fitting.")

        self.t = t
        self.x = x

        if not hasattr(self, 'figures'):
            self.figures = {}
        self.figures.update({'mcissa': {}})

    def restore_original_data(self):
        '''
        Method to restore original data (x,t) = (x_raw,t_raw)
        '''
        from pycissa.preprocessing.data_cleaning.data_cleaning import detect_censored_data,detect_nan_data
        self.x = self.x_raw
        self.t = self.t_raw
        self.censored,num_censored = detect_censored_data(self.x.flatten())  #if we restore the data we must check if the restored data is censored again...
        self.isnan = detect_nan_data(self.x.flatten())

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
        Function to find and replace upper and lower censored data in the multivariate input array x.

        Parameters
        ----------
        replace_type : str, optional
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
        self : MCissa
        '''
        if self.censored:
            from pycissa.preprocessing.data_cleaning.data_cleaning import _fix_censored_data, detect_nan_data, detect_censored_data

            # Since self.x is a 2D array, we can iterate over its columns and process each variable
            new_x = np.empty_like(self.x, dtype=np.float64)
            censoring = np.empty_like(self.x, dtype=object)

            for m in range(self.x.shape[1]):
                col_fixed, col_censoring = _fix_censored_data(self.x[:, m],
                                         replacement_type = replace_type,
                                         lower_multiplier = lower_multiplier,
                                         upper_multiplier = upper_multiplier,
                                         default_value_lower = default_value_lower,
                                         default_value_upper = default_value_upper,
                                         hicensor_lower = hicensor_lower,
                                         hicensor_upper = hicensor_upper,)
                new_x[:, m] = col_fixed
                censoring[:, m] = col_censoring

            self.x = new_x
            self.censoring = censoring

            self.isnan = detect_nan_data(self.x.flatten())
            self.censored,_ = detect_censored_data(self.x.flatten())
            self.information_text += f'''
            ------------------------------------------------------
            Censored data replaced
            '''

        else: warnings.warn("WARNING: No censored data detected. Returning unchanged data.")

        return self

    def post_group_manual(self,
                                I:                         int|float|dict,
                                season_length:             int = 1,
                                cycle_length:              list = [1.5,8],
                                include_noise:             bool = True,):
        '''
        GROUP - Manual Grouping step of M-CiSSA.
        '''

        from pycissa.postprocessing.grouping.m_grouping_functions import m_group
        necessary_attributes = ["Z_stacked", "psd", "L", "results"]
        for attr_i in necessary_attributes:
            if not hasattr(self, attr_i):
                raise ValueError(f"Attribute {attr_i} does not appear to exist in the class. Please run the mcissa fit method first.")

        rc, sh, kg, psd_sh = m_group(self.Z_stacked,
                         self.psd,
                         I,
                         season_length=season_length,
                         cycle_length=cycle_length,
                         include_noise=include_noise
                         )

        self.results['mcissa']['manual'] = {}
        self.results['mcissa']['manual']['rc'] = rc
        self.results['mcissa']['manual']['sh'] = sh
        self.results['mcissa']['manual']['kg'] = kg
        self.results['mcissa']['manual']['psd_sh'] = psd_sh

        return self

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
        self : MCissa object
            DESCRIPTION: MCissa object
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
        self : MCissa object
        '''

        if version == 'date':
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

        # Count properly for numpy array logic
        added_count = np.sum(self.added_times == True) if self.added_times is not None else 0

        self.information_text += f'''
        ------------------------------------------------------
        {added_count} number of samples missing in the time series to ensure it is approximately evenly spaced.
        '''

        return self

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
                  component_selection_method: str = 'drop_smallest_n',
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
        Multivariate implementation of gap filling. Fills gaps using joint multivariate CiSSA gap filling.
        '''

        if convergence is None:
            convergence = ['value', 0.01 * np.nanmin(self.x)]

        # Variables to store results across all M channels
        M = self.x.shape[1]

        if not hasattr(self, 'figures'):
            self.figures = {}
        if 'mcissa' not in self.figures:
            self.figures['mcissa'] = {}

        self.figures['mcissa']['figure_gap_fill_error'] = []
        self.figures['mcissa']['figure_gap_fill'] = []

        if self.censored:
            raise ValueError("Censored data detected. Please run pre_fix_censored_data before fitting.")

        from pycissa.preprocessing.gap_fill.gap_filling import m_fill_timeseries_gaps
        res = m_fill_timeseries_gaps(self.t, self.x, L=L, convergence=convergence, extension_type=extension_type,
                                    multi_thread_run=multi_thread_run, initial_guess=initial_guess, outliers=outliers,
                                    estimate_error=estimate_error, test_number=test_number, test_repeats=test_repeats,
                                    z_value=z_value, component_selection_method=component_selection_method,
                                    eigenvalue_proportion=eigenvalue_proportion, number_of_groups_to_drop=number_of_groups_to_drop,
                                    data_per_unit_period=data_per_unit_period, max_iter=max_iter, verbose=verbose, **kwargs)

        x_ca, err_est, err_est_perc, err_rmse, err_rmse_perc, orig_pts, imp_pts, fig_err, fig_ts = res
        self.x = x_ca

        # Storing these simply since m_fill_timeseries_gaps isn't tracking individual channels the exact same way
        self.gap_fill_error_estimates = err_est
        self.gap_fill_error_estimates_percentage = err_est_perc
        self.gap_fill_error_rmse = err_rmse
        self.gap_fill_error_rmse_percentage = err_rmse_perc
        self.gap_fill_original_points = orig_pts
        self.gap_fill_imputed_points = imp_pts

        self.figures['mcissa']['figure_gap_fill_error'].append(fig_err)
        self.figures['mcissa']['figure_gap_fill'].append(fig_ts)

        from pycissa.preprocessing.data_cleaning.data_cleaning import detect_nan_data
        self.isnan = detect_nan_data(self.x)

        avg_rmse = self.gap_fill_error_rmse
        avg_rmse_perc = self.gap_fill_error_rmse_percentage

        self.information_text += f'''
        ------------------------------------------------------
        Joint Gap fill RMSE  : {avg_rmse}
        Joint Gap fill % RMSE: {avg_rmse_perc}
        '''

        return self

    def pre_fill_uneven_timeseries(self,
                                   L_values: list[int],
                                   dt: float,
                                   gap_threshold: float,
                                   eps_values: list[float] = None,
                                   interp_method: str = 'cubic',
                                   optimization_metric: str = 'rmse',
                                   r2_warning_threshold: float = 0.5,
                                   plot: bool = True,
                                   **kwargs):
        '''
        Multivariate implementation of uneven gap filling. Iterates over each channel
        independently to fill gaps using univariate uneven gap filling.
        '''
        from pycissa.preprocessing.gap_fill.uneven_gap_filling import fill_uneven_timeseries
        import warnings

        M = self.x.shape[1]
        t_uneven = self.t.copy()

        best_L_list = []
        best_eps_list = []
        rmse_list = []
        r2_list = []
        ccc_list = []

        # Will replace self.t with the even grid and self.x with filled values
        t_even_final = None
        x_even_filled_full = None

        for m in range(M):
            x_m = self.x[:, m]

            # If all nan, skip gracefully
            if np.all(np.isnan(x_m)):
                continue

            res = fill_uneven_timeseries(t=t_uneven, x=x_m, L_values=L_values, dt=dt, gap_threshold=gap_threshold,
                                        eps_values=eps_values, interp_method=interp_method,
                                        optimization_metric=optimization_metric,
                                        r2_warning_threshold=r2_warning_threshold, plot=plot, **kwargs)

            if t_even_final is None:
                t_even_final = res['t_even']
                x_even_filled_full = np.zeros((len(t_even_final), M))

            x_even_filled_full[:, m] = res['x_even_filled']

            best_L_list.append(res['best_L'])
            best_eps_list.append(res['best_eps'])
            rmse_list.append(res['rmse'])
            r2_list.append(res['r2'])
            ccc_list.append(res['ccc'])

            if plot and 'fig' in res:
                if 'mcissa' not in self.figures:
                    self.figures['mcissa'] = {}
                if 'figure_uneven_gap_fill' not in self.figures['mcissa']:
                    self.figures['mcissa']['figure_uneven_gap_fill'] = []
                self.figures['mcissa']['figure_uneven_gap_fill'].append(res['fig'])

        self.t = t_even_final
        self.x = x_even_filled_full

        from pycissa.preprocessing.data_cleaning.data_cleaning import detect_nan_data
        self.isnan = detect_nan_data(self.x)

        self.uneven_gap_fill_best_L = best_L_list
        self.uneven_gap_fill_rmse = rmse_list
        self.uneven_gap_fill_r2 = r2_list

        self.information_text += f'''
        ------------------------------------------------------
        Uneven gap fill average RMSE across channels: {np.nanmean(rmse_list)}
        Uneven gap fill average R2 across channels: {np.nanmean(r2_list)}
        '''

        return self

    def auto_denoise(self,
                     L:             int = None,
                     plot_denoised: bool = True,
                     **kwargs):
        '''
        Function to automatically denoise a multivariate time series using M-CiSSA.
        '''
        if not L:
            L = int(np.floor(len(self.x)/2))

        _ = self.auto_fix_censoring_nan(L,**kwargs)

        _ = self.fit(
                L,
                extension_type = kwargs.get('extension_type','AR_LR'),
                extend_left = kwargs.get('extend_left',True),
                extend_right = kwargs.get('extend_right',True))

        grouping_type = kwargs.get('grouping_type', 'smallest_proportion')
        if grouping_type == 'monte_carlo':
            _ = self.post_run_monte_carlo_analysis(
                                         alpha                    = kwargs.get('alpha',0.05),
                                         K_surrogates             = kwargs.get('K_surrogates',1),
                                         surrogates               = kwargs.get('surrogates','random_permutation'),
                                         seed                     = kwargs.get('seed',None),
                                         sided_test               = kwargs.get('sided_test','one sided'),
                                         remove_trend             = kwargs.get('remove_trend',True),
                                         trend_always_significant = kwargs.get('trend_always_significant',True),
                                         )

        _ = self.post_group_components(
                                      grouping_type            = grouping_type,
                                      eigenvalue_proportion    = kwargs.get('eigenvalue_proportion',0.9),
                                      number_of_groups_to_drop = kwargs.get('number_of_groups_to_drop',5),
                                      include_trend            = kwargs.get('include_trend',True),
                                      plot_result              = kwargs.get('plot_result',False))

        self.x_denoised = self.x_trend + self.x_periodic

        if plot_denoised:
            pass

        return self

    def auto_detrend(self,
                     L:           int = None,
                     plot_result: bool = True,
                     **kwargs):
        '''
        Function to automatically detrend a M-CiSSA signal.
        '''
        if not L:
            L = int(np.floor(len(self.x)/2))

        _ = self.auto_fix_censoring_nan(L,**kwargs)

        _ = self.fit(
                L,
                extension_type = kwargs.get('extension_type','AR_LR'),
                extend_left = kwargs.get('extend_left',True),
                extend_right = kwargs.get('extend_right',True))

        from pycissa.postprocessing.grouping.m_grouping_functions import m_group
        nft = self.Z_stacked.shape[2]
        I = {'trend'    :[0],
             'detrended':[x for x in range(1, int(nft))]}
        rc, sh, kg, psd_sh = m_group(self.Z_stacked, self.psd, I)

        self.x_trend = rc['trend']
        self.x_detrended = rc['detrended']

        if plot_result:
            pass

        return self

    def auto_cissa(self,
                   L: int = None,
                   **kwargs):
        '''
        AUTO-MCISSA!
        '''
        if not L:
            L = int(np.floor(len(self.x)/2))

        print('Checking for censored or nan data...')
        _ = self.auto_fix_censoring_nan(L,**kwargs)

        print('RUNNING M-CISSA!')
        _ = self.fit(
                L,
                extension_type = kwargs.get('extension_type','AR_LR'),
                extend_left = kwargs.get('extend_left',True),
                extend_right = kwargs.get('extend_right',True))

        print('Performing monte-carlo significance analysis...')
        if kwargs.get('grouping_type','smallest_proportion')=='monte_carlo':
            _ = self.post_run_monte_carlo_analysis(
                                         alpha                    = kwargs.get('alpha',0.05),
                                         K_surrogates             = kwargs.get('K_surrogates',1),
                                         surrogates               = kwargs.get('surrogates','random_permutation'),
                                         seed                     = kwargs.get('seed',None),
                                         sided_test               = kwargs.get('sided_test','one sided'),
                                         remove_trend             = kwargs.get('remove_trend',True),
                                         trend_always_significant = kwargs.get('trend_always_significant',True),
                                         )

        print('Grouping components...')
        grouping_type = kwargs.get('grouping_type', 'smallest_proportion')
        _ = self.post_group_components(
                                      grouping_type            = grouping_type,
                                      eigenvalue_proportion    = kwargs.get('eigenvalue_proportion',0.9),
                                      number_of_groups_to_drop = kwargs.get('number_of_groups_to_drop',5),
                                      include_trend            = kwargs.get('include_trend',True),
                                      plot_result              = kwargs.get('plot_result',False))
        print("Auto M-CiSSA Complete!")
        return self

    def auto_blind_source_separation(self,
                                     L: int = None,
                                     main_index: int = 0,
                                     alpha: float = 0.05,
                                     **kwargs):
        '''
        Automatically remove the influence of additional reference series from a main series.
        Uses M-CiSSA Monte Carlo surrogate testing to identify components that are statistically significant in the reference channels.
        These significant "influence" components are subtracted from the main signal.
        '''
        if not L:
            L = int(np.floor(len(self.x)/2))

        print('Checking for censored or nan data...')
        _ = self.auto_fix_censoring_nan(L,**kwargs)

        print('RUNNING M-CISSA!')
        _ = self.fit(
                L,
                extension_type = kwargs.get('extension_type','AR_LR'),
                extend_left = kwargs.get('extend_left',True),
                extend_right = kwargs.get('extend_right',True))

        print('Performing reference-channel monte-carlo significance analysis...')
        from pycissa.postprocessing.monte_carlo.m_montecarlo_reference import run_m_monte_carlo_reference_test

        surrogates = kwargs.get('surrogates','random_permutation')
        reference_indices = [i for i in range(self.x.shape[1]) if i != main_index]

        mc_results = run_m_monte_carlo_reference_test(x = self.x,
                             L = self.L,
                             psd = self.psd,
                             results = self.results.get('mcissa'),
                             reference_indices = reference_indices,
                             alpha = alpha,
                             K_surrogates = kwargs.get('K_surrogates',1),
                             surrogates = surrogates,
                             seed = kwargs.get('seed',None),
                             sided_test = kwargs.get('sided_test','one sided'),
                             remove_trend = kwargs.get('remove_trend',True),
                             trend_always_significant = kwargs.get('trend_always_significant',False),
                             extension_type = self.extension_type,
                             extend_left = True,
                             extend_right = True,
                                 )


        self.results['mcissa']['model parameters'].update({'monte_carlo_surrogate_type': surrogates})
        self.results['mcissa']['model parameters'].update({'monte_carlo_alpha': alpha})

        variance_threshold = kwargs.get('variance_threshold', 0.25)

        M = self.x.shape[1]
        T_len = self.x.shape[0]
        nft = self.Z_stacked.shape[2]

        self.x_cleaned = np.zeros(T_len)
        self.x_influence = np.zeros(T_len)

        self.x_cleaned_components = np.zeros((T_len, nft))
        self.x_influence_components = np.zeros((T_len, nft))
        self.blind_source_separation_main_index = main_index

        # M-CiSSA provides M subcomponents at every frequency.
        # We drop the specific spatial eigenvectors (subcomponents) that are heavily driven by the references,
        # allowing separation of sources sharing the exact same frequency!
        for k in range(nft):
            for m in range(M):
                subcomp_power_main = np.var(self.Zs[main_index][:T_len, m, k])
                subcomp_power_refs = np.sum([np.var(self.Zs[ref][:T_len, m, k]) for ref in reference_indices])
                total_subcomp_power = subcomp_power_main + subcomp_power_refs

                if total_subcomp_power > 0:
                    p_ratio = subcomp_power_refs / total_subcomp_power
                else:
                    p_ratio = 0.0

                # We optionally incorporate the Monte Carlo pass flag for frequency k.
                mc_pass = False
                for component_j in mc_results.get('components', {}):
                    if mc_results.get('components').get(component_j).get('array_position') == k:
                        mc_pass = mc_results.get('components').get(component_j).get('monte_carlo', {}).get(surrogates, {}).get('alpha', {}).get(alpha, {}).get('pass', False)
                        break

                if mc_pass and p_ratio > variance_threshold:
                    self.x_influence += self.Zs[main_index][:T_len, m, k]
                    self.x_influence_components[:, k] += self.Zs[main_index][:T_len, m, k]
                else:
                    self.x_cleaned += self.Zs[main_index][:T_len, m, k]
                    self.x_cleaned_components[:, k] += self.Zs[main_index][:T_len, m, k]

        print("Auto Blind Source Separation Complete!")
        return self

    def auto_cissa_classic(self,
                   I:                         int|float|dict,
                   L: int = None,
                   season_length:             int = 1,
                   cycle_length:              list = [1.5,8],
                   **kwargs):
        '''
        This version of auto_cissa (classic) implements manual grouping rules natively.
        '''
        if not L:
            L = int(np.floor(len(self.x)/2))

        print('Checking for censored or nan data...')
        _ = self.auto_fix_censoring_nan(L,**kwargs)

        self.fit(L=L,
                 extension_type   = kwargs.get('extension_type','AR_LR'),
                 extend_left = kwargs.get('extend_left',True),
                 extend_right = kwargs.get('extend_right',True)
                 )

        self.post_group_manual(I=I,
                               season_length  = kwargs.get('season_length',1),
                               cycle_length   = kwargs.get('cycle_length',[1.5,8]),
                               include_noise  = kwargs.get('include_noise',True),
                               )

        rc = self.results['mcissa']['manual']['rc']
        if type(I) == int or type(I) == float:
            if ((I-np.floor(I))==0) & (I>0):
                self.x_trend = rc['trend']
                self.x_seasonality = rc['seasonality']
                self.x_long_term_cycle = rc['long term cycle']
                self.x_noise = rc['noise']
        else:
            self.x_reconstructed = rc

        return self

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
        import warnings
        import numpy as np

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
            warnings.warn("NaN data detected. Running pre_fill_gaps to fix...")
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
                          component_selection_method = kwargs.get('component_selection_method','drop_smallest_n'), # Modified to drop_smallest_n to match M-CiSSA logic easier initially
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

    def fit(self, L: int, extension_type: str = 'AR_LR', extend_left: bool = True, extend_right: bool = True):
        """
        Function to fit M-CiSSA to a multivariate timeseries.
        """
        target_dtype = np.float32 if self.use_32_bit else np.float64
        try:
            self.x = np.asarray(self.x, dtype=target_dtype)
        except ValueError:
            raise ValueError("All elements in the input array 'x' must be numeric or convertible to numeric type before fitting.")

        from pycissa.processing.matrix_operations.m_matrix_operations import run_mcissa

        self.Z_stacked, self.psd, self.Zs = run_mcissa(
            self.x, L,
            extension_type=extension_type,
            extend_left=extend_left,
            extend_right=extend_right
        )

        # We can store Z components like univariate CiSSA.
        # self.Z_stacked has shape (T, M, nft)

        self.L = L
        self.extension_type = extension_type

        # generate initial results dictionary
        from pycissa.utilities.generate_cissa_result_dictionary import generate_m_results_dictionary
        from pycissa.postprocessing.grouping.grouping_functions import generate_grouping

        if not hasattr(self, 'results'):
            self.results = generate_m_results_dictionary(self.Z_stacked, self.psd, L, cissa_type='mcissa')
        else:
            self.results.update(generate_m_results_dictionary(self.Z_stacked, self.psd, L, cissa_type='mcissa'))

        self.frequencies = generate_grouping(np.zeros(L), L, trend=True)

        results = self.results
        results.get('mcissa').setdefault('model parameters', {})
        results.get('mcissa').setdefault('noise component tests', {})
        results.get('mcissa').setdefault('fractal scaling results', {})
        results.get('mcissa').get('model parameters').update({
            'extension_type': extension_type,
            'L': L,
        })
        self.results = results

        return self


    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def post_run_frequency_time_analysis(self,
                                    data_per_period:    int,
                                    channel_index:      int = 0,
                                    use_cleaned:        bool = False,
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
        Function to generate frequency-time and period-time matrices and figures for M-CiSSA.
        '''
        from pycissa.postprocessing.frequency_time.frequency_time import _run_frequency_time_analysis

        necessary_attributes = ["Z_stacked", "psd", "t", "L", "results"]
        for attr_i in necessary_attributes:
            if not hasattr(self, attr_i): raise ValueError(f"Attribute {attr_i} does not appear to exist in the class. Please run the fit method before running the run_frequency_time_analysis method.")

        if use_cleaned:
            if not hasattr(self, 'x_cleaned_components'):
                raise ValueError("Attribute 'x_cleaned_components' not found. Run auto_blind_source_separation first.")
            if channel_index != getattr(self, 'blind_source_separation_main_index', 0):
                import warnings
                warnings.warn(f"Warning: Using use_cleaned=True but channel_index ({channel_index}) differs from the main index used in BSS ({getattr(self, 'blind_source_separation_main_index', 0)}). Reverting to the BSS main index.")
                channel_index = getattr(self, 'blind_source_separation_main_index', 0)

            # Create a mock Z matrix that mimics the (T, nft) shape expected by the univariate _run_frequency_time_analysis
            # Actually _run_frequency_time_analysis expects Z to be shape (T, nft) natively from CiSSA.
            # `self.x_cleaned_components` has shape `(T, nft)`.
            Z_to_use = self.x_cleaned_components
        else:
            # self.Z_stacked has shape (T, M, nft)
            # We want to extract the components for the specific channel
            Z_to_use = self.Z_stacked[:, channel_index, :]

        # univariate _run_frequency_time_analysis expects psd to be a column vector, so we extract the relevant channel and reshape it
        psd_to_use = self.psd[:, channel_index].reshape(-1, 1)
        self.frequency_list, self.period_list, self.amplitude_matrix, self.power_matrix, self.phase_matrix, _, fig_f, fig_p = _run_frequency_time_analysis(
            Z_to_use, psd_to_use, self.t, self.L,
            data_per_period=data_per_period, period_name=period_name, t_unit=t_unit,
            plot_frequency=plot_frequency, plot_period=plot_period,
            logplot_frequency=logplot_frequency, logplot_period=logplot_period,
            normalise_plots=normalise_plots, height_variable=height_variable, height_unit=height_unit
        )

        suffix = f"_channel_{channel_index}"
        if use_cleaned:
            suffix += "_cleaned"

        if fig_f is not None:
            self.figures.get('mcissa').update({f'figure_frequency_time{suffix}': fig_f})
        if fig_p is not None:
            self.figures.get('mcissa').update({f'figure_period_time{suffix}': fig_p})

        results = self.results
        if 'frequency_time_results' not in results.get('mcissa'):
            results.get('mcissa')['frequency_time_results'] = {}

        results.get('mcissa').get('frequency_time_results').update({
            suffix: {
                'frequency_list'   : self.frequency_list,
                'period_list'      : self.period_list,
                'amplitude_matrix' : self.amplitude_matrix,
                'power_matrix'     : self.power_matrix,
                'phase_matrix'     : self.phase_matrix,
            }
        })

        results.get('mcissa').setdefault('model parameters', {})
        results.get('mcissa').get('model parameters').update({
            'data_per_period'   : data_per_period,
            'period_name'       : period_name,
            't_unit'            : t_unit,
        })

        self.results = results
        self.data_per_period = data_per_period
        self.period_name     = period_name
        self.t_unit          = t_unit
        if plt.get_fignums(): plt.close('all')

        return self

    def post_run_monte_carlo_analysis(self,
                                 alpha:                    float = 0.05,
                                 K_surrogates:             int = 1,
                                 surrogates:               str = 'random_permutation',
                                 seed:                     int|None = None,
                                 sided_test:               str = 'one sided',
                                 remove_trend:             bool = True,
                                 trend_always_significant: bool = True,
                                 plot_figure:              bool = False):
        '''
        Function to run a monte carlo significance test on components of a signal, extracted via M-CiSSA.
        '''
        from pycissa.postprocessing.monte_carlo.m_montecarlo import run_m_monte_carlo_test

        necessary_attributes = ["psd", "L", "results"]
        for attr_i in necessary_attributes:
            if not hasattr(self, attr_i):
                raise ValueError(f"Attribute {attr_i} does not appear to exist in the class. Please run the mcissa fit method first.")

        mc_results, figure_monte_carlo = run_m_monte_carlo_test(x = self.x,
                             L = self.L,
                             psd = self.psd,
                             results = self.results.get('mcissa'),
                             alpha = alpha,
                             K_surrogates = K_surrogates,
                             surrogates = surrogates,
                             seed = seed,
                             sided_test = sided_test,
                             remove_trend = remove_trend,
                             trend_always_significant = trend_always_significant,
                             extension_type = self.extension_type,
                             extend_left = True, # These should ideally match the fit call, assume true for now.
                             extend_right = True,
                             plot_figure = plot_figure
                                 )

        self.results.get('mcissa').update(mc_results)
        self.results['mcissa']['model parameters'].update({'monte_carlo_surrogate_type': surrogates})
        self.results['mcissa']['model parameters'].update({'monte_carlo_alpha': alpha})

        self.information_text += f'''
        ------------------------------------------------------
        MONTE CARLO SIGNIFICANCE TESTING
        '''
        for component_j in self.results.get('mcissa').get('components'):
            if self.results.get('mcissa').get('components').get(component_j).get('monte_carlo').get(surrogates).get('alpha').get(alpha).get('pass'):
                self.information_text += f'''
        Unitless frequency: {component_j} SIGNIFICANT.
                '''
        return self


    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def post_analyse_trend(self,
                      channel_index:     int = 0,
                      use_cleaned:       bool = False,
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
        Method to calculate and generate the trend slope and confidence for the "trend" component of the M-CiSSA results.
        '''
        necessary_attributes = ["t", "results"]
        for attr_i in necessary_attributes:
            if not hasattr(self, attr_i): raise ValueError(f"Attribute {attr_i} does not appear to exist in the class. Please run the fit method first.")

        if use_cleaned:
            if not hasattr(self, 'x_cleaned_components'):
                raise ValueError("Attribute 'x_cleaned_components' not found. Run auto_blind_source_separation first.")
            if channel_index != getattr(self, 'blind_source_separation_main_index', 0):
                import warnings
                warnings.warn(f"Warning: Using use_cleaned=True but channel_index ({channel_index}) differs from the main index used in BSS ({getattr(self, 'blind_source_separation_main_index', 0)}). Reverting to the BSS main index.")
                channel_index = getattr(self, 'blind_source_separation_main_index', 0)

            # Trend is usually at component array position 0
            trend_idx = 0
            for key_j in self.results['mcissa']['components'].keys():
                if key_j == 'trend':
                    trend_idx = self.results['mcissa']['components'][key_j].get('array_position', 0)
                    break

            # The univariate logic normally uses the reconstructed trend from components.
            # In our case, self.x_cleaned_components[:, trend_idx] is the trend component
            trend_data = self.x_cleaned_components[:, trend_idx]
        else:
            if not hasattr(self, 'x_trend'):
                raise ValueError("Attribute 'x_trend' not found. Run post_group_components or auto_detrend first.")
            trend_data = self.x_trend[:, channel_index]

        suffix = f"_channel_{channel_index}"
        if use_cleaned:
            suffix += "_cleaned"

        if trend_type == 'linear':
            from pycissa.postprocessing.trend.trend_functions import trend_linear

            figure_trend, self.trend_slope, self.trend_increasing_probability, self.trend_increasing_probability_text, self.trend_confidence = trend_linear(
                             trend_data,
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
            self.figures.get('mcissa').update({f'figure_trend{suffix}': figure_trend})
        elif trend_type == 'rolling_OLS':
            from pycissa.postprocessing.trend.trend_functions import trend_rolling
            figure_trend, self.trend_slope, self.trend_increasing_probability, self.trend_increasing_probability_text, self.trend_confidence = trend_rolling(
                              trend_data,
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
            self.figures.get('mcissa').update({f'figure_trend{suffix}': figure_trend})
        else:
            raise ValueError(f"Input value trend_type = {trend_type} is incorrect. Please use one of 'linear' or 'rolling_OLS'.")

        results = self.results
        results.get('mcissa').setdefault('trend results', {})
        if suffix not in results.get('mcissa').get('trend results'):
            results.get('mcissa').get('trend results')[suffix] = {}

        results.get('mcissa').get('trend results')[suffix].setdefault(self.trend_type, {})
        results.get('mcissa').get('trend results')[suffix].get(self.trend_type).update({
            'trend_slope'                       : self.trend_slope,
            'trend_increasing_probability'      : self.trend_increasing_probability,
            'trend_increasing_probability_text' : self.trend_increasing_probability_text,
            'trend_confidence'                  : self.trend_confidence
            })
        self.results = results
        if plt.get_fignums(): plt.close('all')
        return self

    def post_group_components(self,
                                 grouping_type:            str = 'monte_carlo',
                                 eigenvalue_proportion:    float = 0.9,
                                 number_of_groups_to_drop: int = 5,
                                 include_trend:            bool = True,
                                 plot_result:              bool = True):
        '''
        Function to group components into trend, periodic, or noise/residual.
        '''
        def combine_m_components(temp_results, group_indices):
            x_grouped = np.zeros(temp_results['components']['trend']['reconstructed_data'].shape)
            for key_j in temp_results['components'].keys():
                if temp_results['components'][key_j]['array_position'] in group_indices:
                    x_grouped += temp_results['components'][key_j]['reconstructed_data']
            return x_grouped

        necessary_attributes = ["psd", "L", "results"]
        for attr_i in necessary_attributes:
            if not hasattr(self, attr_i):
                raise ValueError(f"Attribute {attr_i} does not appear to exist in the class. Please run the mcissa fit method first.")

        if grouping_type == 'monte_carlo':
            if self.results.get('mcissa').get('components').get('trend').get('monte_carlo') is None:
                raise ValueError(f"Please run the post_run_monte_carlo_analysis method before running the post_group_components with grouping_type == 'monte_carlo' or use another grouping type.")
            from pycissa.postprocessing.grouping.m_grouping_functions import m_classify_monte_carlo_non_significant_components
            trend, periodic, noise = m_classify_monte_carlo_non_significant_components(self.results.get('mcissa'))
        elif grouping_type == 'smallest_proportion':
            from pycissa.postprocessing.grouping.m_grouping_functions import m_classify_smallest_proportion_psd
            trend, periodic, noise = m_classify_smallest_proportion_psd(self.Z_stacked,
                                                                       self.psd,
                                                                       self.L,
                                                                       eigenvalue_proportion)
        elif grouping_type == 'smallest_n':
            from pycissa.postprocessing.grouping.m_grouping_functions import m_classify_smallest_n_components
            trend, periodic, noise = m_classify_smallest_n_components(self.Z_stacked,
                                                                     self.psd,
                                                                     self.L,
                                                                     number_of_groups_to_drop,
                                                                     include_trend=include_trend)
        else: raise ValueError(f"Input parameter 'grouping_type' should be one of 'monte_carlo', 'smallest_proportion', or 'smallest_n'. You entered: {grouping_type}.")

        trend_share, periodic_share, noise_share = 0., 0., 0.
        for key_j in self.results['mcissa']['components'].keys():
            index = self.results['mcissa']['components'][key_j]['array_position']
            share = self.results['mcissa']['components'][key_j]['percentage_share_of_psd']
            if index in trend:    trend_share += share
            if index in periodic: periodic_share += share
            if index in noise:    noise_share += share

        self.results['mcissa']['noise component tests'].update({'trend_index': trend})
        self.results['mcissa']['noise component tests'].update({'trend_share': trend_share})
        self.results['mcissa']['noise component tests'].update({'periodic_index': periodic})
        self.results['mcissa']['noise component tests'].update({'periodic_share': periodic_share})
        self.results['mcissa']['noise component tests'].update({'noise_index': noise})
        self.results['mcissa']['noise component tests'].update({'noise_share': noise_share})

        self.x_trend = combine_m_components(self.results['mcissa'], trend)
        self.x_periodic = combine_m_components(self.results['mcissa'], periodic)
        self.x_noise = combine_m_components(self.results['mcissa'], noise)

        if plot_result:
            # We can plot components here using existing utilities if adapted, or skip complex plotting
            pass

        self.information_text += f'''
        ------------------------------------------------------
        COMPONENT VARIANCE
        TREND   : {self.results.get('mcissa').get('noise component tests').get('trend_share')}%
        PERIODIC: {self.results.get('mcissa').get('noise component tests').get('periodic_share')}%
        NOISE   : {self.results.get('mcissa').get('noise component tests').get('noise_share')}%
        '''

        return self

    def plot_components(self, num_components: int = 3, variable_names: list = None, component_names: list = None):
        """
        Groups the components by total variance across all variables and plots the top `num_components`.

        Parameters:
        num_components: int - Number of top components to plot (ordered by variance)
        variable_names: list of str - Names for the variables
        component_names: list of str - Names for the extracted components
        """
        from pycissa.utilities.plotting import plot_m_components

        if not hasattr(self, 'Z_stacked'):
            raise ValueError("Model not fitted. Call fit() before plotting.")

        M = self.Z_stacked.shape[1]
        nft = self.Z_stacked.shape[2]

        # Calculate total variance across all variables for each frequency
        variances = [np.sum([np.var(self.Z_stacked[:, m, i]) for m in range(M)]) for i in range(nft)]

        # Sort indices by descending variance
        sorted_indices = np.argsort(variances)[::-1]

        # Extract the top components
        extracted_components = []
        for i in range(min(num_components, nft)):
            extracted_components.append(self.Z_stacked[:, :, sorted_indices[i]])

        fig = plot_m_components(self.t, self.x, extracted_components, variable_names, component_names)
        self.figures.get('mcissa').update({'figure_components': fig})

        return fig
