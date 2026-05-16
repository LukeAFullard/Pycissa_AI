import numpy as np

def detect_censored_data(x: np.ndarray)-> bool:
    '''
    This function simply checks to see if any of the entriesof the array x are either '<' or '>'.
    Data of this type are usually called censored and is common in environmental data, for example see
    Helsel, D. R. (2005). More than obvious: better methods for interpreting nondetect data. Environmental science & technology, 39(20), 419A-423A.

    Function returns True if any data is censored, otherwise returns False

    Parameters
    ----------
    x : np.ndarray
        DESCRIPTION: Array of input data.

    Returns
    -------
    bool
        DESCRIPTION: If there is censored data detected then returns True, otherwise returns False.

    '''
    censoring = len([y[0] for y in str(x) if y[0] in ['>','<']])
    return censoring > 0,censoring

def detect_nan_data(x: np.ndarray)-> bool:
    '''
    This function simply checks to see if any of the entries of the array x are either nan.

    Function returns True if any data is nan, otherwise returns False

    Parameters
    ----------
    x : np.ndarray
        DESCRIPTION: Array of input data.

    Returns
    -------
    bool
        DESCRIPTION: If there is nan data detected then returns True, otherwise returns False.

    '''
    def is_nan(value):
        try:
            return np.isnan(float(value))
        except (ValueError, TypeError):
            return False
    nan_entries = len([y for y in x if is_nan(y)])
    return nan_entries > 0

###############################################################################
###############################################################################
def _fix_censored_data(x: np.ndarray,
                       replacement_type:     str = 'raw',
                       lower_multiplier: float = 0.5,
                       upper_multiplier: float = 1.1,
                       default_value_lower:    float = 0.,
                       default_value_upper:    float = 0.,
                       hicensor_lower:   bool = False,
                       hicensor_upper:   bool = False) -> tuple[np.ndarray, np.ndarray]:
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


    # if value is numeric, leave as is, if starts with < or >, uncensor and store censoring,
    # if other, return np.nan
    if not replacement_type in ['raw', 'multiple', 'constant']:
        raise ValueError(f'Parameter "replacement_type" should be one of "raw", "multiple", or "constant". Currently  replacement_type = {replacement_type}')
    if not type(lower_multiplier) in [float, int]:
        raise ValueError(f'Parameter "lower_multiplier" = {lower_multiplier} but should be numeric.')
    if not type(upper_multiplier) in [float, int]:
        raise ValueError(f'Parameter "upper_multiplier" = {upper_multiplier} but should be numeric.')
    if not type(default_value_lower) in [float, int]:
        raise ValueError(f'Parameter "default_value_lower" = {default_value_lower} but should be numeric.')
    if not type(default_value_upper) in [float, int]:
        raise ValueError(f'Parameter "default_value_upper" = {default_value_upper} but should be numeric.')


    x_uncensored = []
    x_censoring  = []
    for entry_i in x:
        if type(entry_i) in [int,float]:  #is numeric
            x_uncensored.append(entry_i)
            x_censoring.append(None)
        elif type(entry_i) in [str]:
            try:  #if happens to be a number but stored as a string, convert to float
                x_uncensored.append(float(entry_i))
                x_censoring.append(None)
            except:
                if entry_i == '':
                    x_uncensored.append(np.nan)
                    x_censoring.append(None)
                elif entry_i[0] == '<':  #if first character of string is <, data is left censored
                    try:
                        if replacement_type == 'raw':
                            x_uncensored.append(float(entry_i[1:]))
                        elif replacement_type == 'multiple':
                            x_uncensored.append(lower_multiplier*float(entry_i[1:]))
                        elif replacement_type == 'constant':
                            x_uncensored.append(default_value_lower)
                    except:x_uncensored.append(np.nan)
                    x_censoring.append('<')
                elif entry_i[0] == '>': #if first character of string is >, data is right censored
                    try:
                        if replacement_type == 'raw':
                            x_uncensored.append(float(entry_i[1:]))
                        elif replacement_type == 'multiple':
                            x_uncensored.append(upper_multiplier*float(entry_i[1:]))
                        elif replacement_type == 'constant':
                            x_uncensored.append(default_value_upper)
                    except:x_uncensored.append(np.nan)
                    x_censoring.append('>')
                else: #if first character of the string is not < or > then return nan
                    x_uncensored.append(np.nan)
                    x_censoring.append(None)
        else: #return nan
            x_uncensored.append(np.nan)
            x_censoring.append(None)


    x_uncensored = np.array(x_uncensored, dtype=np.float64)
    x_censoring  = np.array(x_censoring, dtype=object)

    ###########################################################################
    ###########################################################################
    # if hicensor_lower then we replace the lower censored values with the max
    #  censored value
    if hicensor_lower:
        x_uncensored[x_censoring == '<'] = max(x_uncensored[x_censoring == '<'])

    # if hicensor_upper then we replace the upper censored values with the min
    #  censored value
    if hicensor_upper:
        x_uncensored[x_censoring == '>'] = min(x_uncensored[x_censoring == '>'])
    ###########################################################################
    ###########################################################################
    #need to ensure that the final data is float64 and not object
    x_uncensored = np.array(x_uncensored, dtype=np.float64)
    return x_uncensored,x_censoring

###############################################################################
###############################################################################
from datetime import datetime
def _fix_missing_date_samples(t: np.ndarray,
                         x: np.ndarray,
                         start_date:           str|datetime = 'min',
                           years:              int = 0,
                           months:             int = 1,
                           days:               int = 0,
                           hours:              int = 0,
                           minutes:            int = 0,
                           seconds:            int = 0,
                           input_dateformat:   str = '%Y',
                           year_delta:         int = 0,
                           month_delta:        int = 0,
                           day_delta:          int = 14,
                           hour_delta:         int = 0,
                           minute_delta:       int = 0,
                           second_delta:       int = 0,
                           missing_value:      int = np.nan
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Function that finds and corrects misisng dates in the time series.
    Missing dates result in adding a default value "missing_value" into the input data.

    **THIS FUNCTION IS A WORK IN PROGRESS. USE WITH EXTREME CAUTION.**

    Parameters
    ----------
    t : np.ndarray
        DESCRIPTION: array of input times/dates.
    x : np.ndarray
        DESCRIPTION: array of input data.
    start_date : str|datetime
        DESCRIPTION: If = 'min' then the minimum date is used, otherwise the given datetime is taken as the first required time. The default is 'min'.
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
    t_centered : np.ndarray
        DESCRIPTION: array of centered time values. t_centered does not correspond to the t values of final_x, but is an approximation of final_t to help with prediction on a regular time grid.
    '''
    # import datetime
    from dateutil.relativedelta import relativedelta
    import copy
    def add_date_delta(mydate,years,months,days,hours,minutes,seconds,direction):
        if direction == 'subtract':
            years *= -1
            months *= -1
            days *= -1
            hours *= -1
            minutes *= -1
            seconds *= -1
        mydate = mydate + relativedelta(years = years,
                                        months = months,
                                        days = days,
                                        hours = hours,
                                        minutes = minutes,
                                        seconds = seconds)
        return mydate


    #check that sum is not = 0
    if not years+months+days+hours+minutes+seconds>0:
        raise ValueError('One of the input parameters years, months, weeks, days, hours, minutes, seconds must be greater than zero')

    t_ = copy.deepcopy(t)
    t_ = [np.datetime64(dt, 's') for dt in t_]
    t_ = [dt.astype(datetime) for dt in t_]

    new_t = []
    for time_i in t_:
        if type(time_i) in [str]:
            try:new_t.append(datetime.strptime(time_i, input_dateformat))
            except: raise ValueError(f"Input dateformat, {input_dateformat} does not appear to match the format of the provided dates (e.g. {new_t[0]}). Please correct this.")
        elif type(time_i) in [datetime]:
            new_t.append(time_i)
        else:
            new_t.append(time_i)


    min_date = min(new_t)
    max_date = max(new_t)

    if start_date != 'min':
        # in this case, type should be datetime.datetime
        if not type(start_date) == datetime: raise TypeError(f"If start_date is not 'min' then a datetime object must be provided. Current types is {type(start_date)}")

        #make the start date the provided date
        min_date = start_date
        #now remove all times that are less than the start date
        lower_start_date = add_date_delta(min_date,year_delta,month_delta,day_delta,hour_delta,minute_delta,second_delta,'subtract')
        ok_times = [True if x >= lower_start_date else False for x in new_t]
        x = x[ok_times]
        new_t = [x for x in new_t if x >= lower_start_date ]


    # mydate = mydate.replace(day=mydate.day+1)

    all_dates = []
    centered_dates = []
    all_x = []
    x_missing = []
    current_date = min_date
    for time_i, x_i in zip(new_t,x):

        lower_date = add_date_delta(current_date,year_delta,month_delta,day_delta,hour_delta,minute_delta,second_delta,'subtract')
        upper_date = add_date_delta(current_date,year_delta,month_delta,day_delta,hour_delta,minute_delta,second_delta,'add')
        if (time_i >= lower_date) & (time_i <= upper_date):
            # Here date is within the acceptable range
            all_dates.append(time_i)
            centered_dates.append(current_date)
            all_x.append(x_i)
            x_missing.append(None)
            current_date = add_date_delta(current_date,years,months,days,hours,minutes,seconds,'add')
        else:
            stop_date = add_date_delta(time_i,year_delta,month_delta,day_delta,hour_delta,minute_delta,second_delta,'add')
            while current_date <= stop_date:
                all_dates.append(current_date)
                centered_dates.append(current_date)
                if (time_i >= lower_date) & (time_i <= upper_date):
                    all_x.append(x_i)
                    x_missing.append(None)
                else:
                    all_x.append(missing_value)
                    x_missing.append(True)
                current_date = add_date_delta(current_date,years,months,days,hours,minutes,seconds,'add')
                lower_date = add_date_delta(current_date,year_delta,month_delta,day_delta,hour_delta,minute_delta,second_delta,'subtract')
                upper_date = add_date_delta(current_date,year_delta,month_delta,day_delta,hour_delta,minute_delta,second_delta,'add')

    final_t    =  np.array(all_dates, dtype=object)
    t_centered =  np.array(centered_dates, dtype=object)
    final_x    =  np.array(all_x,     dtype=np.float64)
    x_missing  =  np.array(x_missing, dtype=np.float64)

    return final_t, final_x, x_missing,t_centered


def _fix_missing_numeric_samples(t: np.ndarray,
                           x: np.ndarray,
                           t_step:             int|float = 1.,
                           wiggleroom:         int|float = 0.99,
                           missing_value:      int = np.nan
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Function that finds and corrects misisng dates in the time series.
    Missing dates result in adding a default value "missing_value" into the input data.

    **THIS FUNCTION IS A WORK IN PROGRESS. USE WITH EXTREME CAUTION.**

    Parameters
    ----------
    t : np.ndarray
        DESCRIPTION: array of input times/dates.
    x : np.ndarray
        DESCRIPTION: array of input data.
    t_step : int|float, optional
        DESCRIPTION: numeric value of the time step. The default is 1.
    wiggleroom : int|float, optional
        DESCRIPTION: Numeric value for the 'wiggle room' associated with a tolerance tolerance interval around the desired timestep. If the time is within the "wiggleroom", then the time is OK. For example, if we have a time step of 2 and the wiggle room is 0.2, then a series of times 2,4,6,7.9,10,... would be OK, but 2,4,6,7.7,10 would not and would correct the time value to 2,4,6,8,10. The default is 0.99.
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
    t_centered : np.ndarray
        DESCRIPTION: array of centered time values. t_centered does not correspond to the t values of final_x, but is an approximation of final_t to help with prediction on a regular time grid.

    '''
    # import datetime
    from datetime import datetime
    from dateutil.relativedelta import relativedelta
    import copy
    def add_date_delta(mystep,tstep,direction):
        if direction == 'subtract':
            tstep *= -1.
        mystep = mystep + tstep
        return mystep


    #check that t_step is not = 0
    if not t_step>0:
        raise ValueError('t_step must be greater than zero')

    new_t = copy.deepcopy(t)

    min_t = min(new_t)
    max_t = max(new_t)

    # mydate = mydate.replace(day=mydate.day+1)

    all_t = []
    t_centered = []
    all_x = []
    x_missing = []
    current_t = min_t
    for time_i, x_i in zip(new_t,x):

        lower_t = add_date_delta(current_t,wiggleroom,'subtract')
        upper_t = add_date_delta(current_t,wiggleroom,'add')
        if (time_i >= lower_t) & (time_i <= upper_t):
            # Here date is within the acceptable range
            all_t.append(time_i)
            t_centered.append(current_t)
            all_x.append(x_i)
            x_missing.append(None)
            current_t = add_date_delta(current_t,t_step,'add')
        else:
            stop_t = add_date_delta(time_i,wiggleroom,'add')
            while current_t <= stop_t:
                all_t.append(current_t)
                t_centered.append(current_t)
                if (time_i >= lower_t) & (time_i <= upper_t):
                    all_x.append(x_i)
                    x_missing.append(None)
                else:
                    all_x.append(missing_value)
                    x_missing.append(True)
                current_t = add_date_delta(current_t,t_step,'add')
                lower_t = add_date_delta(current_t,wiggleroom,'subtract')
                upper_t = add_date_delta(current_t,wiggleroom,'add')

    final_t    =  np.array(all_t, dtype=np.float64)
    t_centered =  np.array(t_centered, dtype=np.float64)
    final_x    =  np.array(all_x,     dtype=np.float64)
    x_missing  =  np.array(x_missing, dtype=np.float64)

    return final_t, final_x, x_missing, t_centered
