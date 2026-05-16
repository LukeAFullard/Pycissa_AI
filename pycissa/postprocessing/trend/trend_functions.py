import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
import datetime
import copy
min_width = 720
min_height = 570

confidence_colour_map = {
    'Highly unlikely'     :   '#ff0000',
    'Very unlikely'       :   '#f64d0a',
    'Unlikely'            :   '#ff9933',
    'As likely as not'    :   '#ffff00',
    'Likely'              :   '#92d050',
    'Very likely'         :   '#00b0f0',
    'Highly likely'       :   '#0070c0'
    }

confidence_increasing_names = ['Highly unlikely',
                               'Very unlikely',
                               'Unlikely',
                               'As likely as not',
                               'Likely',
                               'Very likely',
                               'Highly likely']
confidence_increasing_bins = [0,0.050001,0.10001,0.33001, 0.67001, 0.90001, 0.95001, 1]
###############################################################################
###############################################################################
###############################################################################
def plot_trend(Y:               np.ndarray,
               t:               np.ndarray,
               slopes:          np.ndarray,
               increasing_prob: np.ndarray,
               increasing_text: np.ndarray,
               include_data:    bool = True,
               legend_loc:      int = 2,
               shade_area:      bool = False,
               xaxis_rotation:  float = 270,
               t_unit:          str = '',
               Y_unit:          str = '',
               timestep_unit:   str = ''):
    '''
    Function to plot a trend figure using rolling Ordinary Least Squares.

    Parameters
    ----------
    Y : np.ndarray
        DESCRIPTION: Input array of the original data.
    t : np.ndarray
        DESCRIPTION: Array of input times.
    slopes : np.ndarray
        DESCRIPTION: Array of rolling trends slopes
    increasing_prob : np.ndarray
        DESCRIPTION: Numeric probability of increasing slope associated with each of the slopes in the input variable "slopes".
    increasing_text : np.ndarray
        DESCRIPTION: Text classification of the increasing probabilities associated with the input variable increasing_prob
    include_data : bool, optional
        DESCRIPTION: Whether to include the original time-series in the plot or not. The default is True.
    legend_loc : int, optional
        DESCRIPTION: Location of the legend. The default is 2.
    shade_area : bool, optional
        DESCRIPTION: Whether to shade below the trend or not. The default is False.
    xaxis_rotation : float, optional
        DESCRIPTION: Angle (degrees) to control of the x-axis ticks. The default is 270.
    t_unit : str, optional
        DESCRIPTION: Time unit. Not required if time is a datetime. The default is ''.
    Y_unit : str, optional
        DESCRIPTION: Data unit. The default is ''.
    timestep_unit : str, optional
        DESCRIPTION: Timestep unit (e.g. seconds, days, years). The default is ''.

    Returns
    -------
    fig : matplotlib.figure
        DESCRIPTION: Trend figure.

    '''

    my_colours = np.array([confidence_colour_map.get(x,'#FFFFFF') for x in increasing_text])

    if isinstance(t[-1],(np.datetime64,datetime.datetime)):
        t_unit = 'Date'

    fig, ax = plt.subplots(2)
    ax[0].scatter(t, slopes, c=my_colours, alpha=0.5)
    if include_data:
        ax[1].plot(t, Y,'k')
    ax[0].grid(True)

    #add legend
    my_patches = []
    for increasing_text_i,colour_i in confidence_colour_map.items():
        my_patches.append(mpatches.Patch(color=colour_i, label=increasing_text_i))
    ax[0].legend(handles=my_patches,title='Probability of an increasing trend', loc = legend_loc, bbox_to_anchor=(1.04, 1))
    # fig.tight_layout()

    if shade_area:
        for increasing_text_i,colour_i in confidence_colour_map.items():
            ax[0].fill_between(t, slopes, slopes*0, where=my_colours == colour_i, color=colour_i, alpha=0.3,
                     interpolate=True)
    plt.xticks(rotation=xaxis_rotation)
    ax[1].set_xlabel(t_unit)
    ax[0].set_ylabel("Trend Slope \n (" + Y_unit + '/' + timestep_unit + ")")
    ax[1].set_ylabel("Trend component \n (" + Y_unit +")")
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
def plot_linear_trend(Y:               np.ndarray,
                      t:               np.ndarray,
                      timestep:        float,
                      slopes:          float,
                      intercept:       float,
                      increasing_prob: float,
                      increasing_text: str,
                      include_data:    bool = True,
                      legend_loc:      int = 2,
                      shade_area:      bool = False,
                      xaxis_rotation:  float = 270,
                      t_unit:          str = '',
                      Y_unit:          str = '',
                      timestep_unit:   str = ''):
    '''
    Function to plot a linear trend figure.

    Parameters
    ----------
    Y : np.ndarray
        DESCRIPTION: Input array of the original data.
    t : np.ndarray
        DESCRIPTION: Array of input times.
    timestep : float
        DESCRIPTION: Numeric timestep size in t_unit units.
    slopes : float
        DESCRIPTION: Linear trend slope.
    intercept : float
        DESCRIPTION: Linear trend intercept.
    increasing_prob : float
        DESCRIPTION: Probability of an increasing linear trend.
    increasing_text : str
        DESCRIPTION: Text classification of increasing_prob
    include_data : bool, optional
        DESCRIPTION: Whether to include the original time-series in the plot or not. The default is True.
    legend_loc : int, optional
        DESCRIPTION: Location of the legend. The default is 2.
    shade_area : bool, optional
        DESCRIPTION: Whether to shade below the trend or not. The default is False.
    xaxis_rotation : float, optional
        DESCRIPTION: Angle (degrees) to control of the x-axis ticks. The default is 270.
    t_unit : str, optional
        DESCRIPTION: Time unit. Not required if time is a datetime. The default is ''.
    Y_unit : str, optional
        DESCRIPTION: Data unit. The default is ''.
    timestep_unit : str, optional
        DESCRIPTION: Timestep unit (e.g. seconds, days, years). The default is ''.

    Returns
    -------
    fig : matplotlib.figure
        DESCRIPTION: Trend figure.

    '''


    my_colours = np.array([confidence_colour_map.get(x,'#FFFFFF') for x in increasing_text])
    if isinstance(t[-1],(np.datetime64,datetime.datetime)):
        t_unit = 'Date'
    t_ = copy.deepcopy(t)
    if isinstance(t[-1],np.datetime64):
        #convert to datetime.datetime
        t_ = t = [x.astype('datetime64[s]').astype(datetime.datetime) for x in t]

    if isinstance(t[-1],datetime.date):
        t = [datetime.datetime.combine(d, datetime.datetime.min.time()) for d in t]

    if isinstance(t[0],np.ndarray):
        t_ = np.array([dt[0].astype(datetime.datetime) for dt in t])

    if isinstance(t_[-1],datetime.datetime):
        t_ = np.array([dt.timestamp() for dt in t])
        t_ -= t_[0]
        #divide by timestep
        if timestep is not None:
            t_ = t_/timestep
            if timestep_unit is None:
                timestep_unit = str(timestep) + ' seconds'
        else:
            timestep_unit = 'second'

    fig, ax = plt.subplots()
    ax.scatter(t, slopes*t_+intercept, c=my_colours, alpha=0.5)
    if include_data:
        ax.plot(t, Y,'k')

    #add legend
    my_patches = []
    for increasing_text_i,colour_i in confidence_colour_map.items():
        if increasing_text_i == increasing_text[-1]:
            my_patches.append(mpatches.Patch(color=colour_i, label=increasing_text_i))
    ax.legend(handles=my_patches,title='Probability of an increasing trend', loc = legend_loc, bbox_to_anchor=(1.04, 1))

    # if shade_area:
    #     ax.fill_between(t, slopes*t_+intercept, t_*0, color=my_colours, alpha=0.3,
    #              interpolate=True)
    if shade_area:
        for increasing_text_i,colour_i in confidence_colour_map.items():

            ax.fill_between(t, slopes*t_+intercept, t_*0, where=my_colours == colour_i, color=colour_i, alpha=0.3,
                     interpolate=True)

    ax.grid(True)
    fig.tight_layout()
    plt.xticks(rotation=xaxis_rotation)
    ax.set_xlabel(t_unit)
    ax.set_ylabel(Y_unit)
    ax.set_title(f"Trend slope = {slopes} {Y_unit}/{timestep_unit}")
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
def probabiliy_increasing_trend(confidence: np.ndarray|float,
                                rolling:    bool = False,
                                target:     int = 0):
    '''
    Function to generate classification of trend confidence.
    Allows the user to set a target trend slope and outputs the probability we are exceeding that trend slope.

    Parameters
    ----------
    confidence : dict
        DESCRIPTION: dictionary/numeric value of slopes.
    rolling : bool, optional
        DESCRIPTION. Flag, if True we are using a single linear trend, if False, using Rolling Ordinary Least Squares. The default is False.
    target : int, optional
        DESCRIPTION. Target slope. The default is 0.

    Returns
    -------
    increasing_prob, np.ndarray|float
        DESCRIPTION: Probability of exceeding the target slope.
    increasing_text : np.ndarray|str
        DESCRIPTION: Text classification of the probability of exceeding the target slope.

    '''
    def probability_of_exceeding_target_slope(confidence_dict,confidence_increasing_bins,confidence_increasing_names,target=0):

        if any(np.isnan(val) for val in confidence_dict.values()):
            return np.nan,np.nan

        if confidence_dict.get(0.5) == 0:
            # print('case0')
            increasing_text = 'As likely as not'
            increasing_prob = 0.5
        elif min(confidence_dict.values()) > 0:
            # print('case1')
            increasing_text = confidence_increasing_names[-1]
            increasing_prob = f'>{max(confidence_dict.keys()):.3f}'
        elif max(confidence_dict.values()) < 0:
            # print('case2')
            increasing_text = confidence_increasing_names[0]
            increasing_prob = f'<{min(confidence_dict.keys()):.3f}'
        else:
            # print('case3')
            increasing_prob = np.interp(0,
                                        np.flip(np.fromiter(confidence_dict.values(), dtype=float)),
                                        np.flip(np.fromiter(confidence_dict.keys(), dtype=float))
                                        )
            increasing_text = confidence_increasing_names[np.digitize(increasing_prob,confidence_increasing_bins, right = False)-1]
        return increasing_prob,increasing_text

    if not rolling:
        #here we have linear fit (i.e. not rolling ols)
        increasing_prob,increasing_text = probability_of_exceeding_target_slope(confidence,confidence_increasing_bins,confidence_increasing_names,target=target)

        increasing_prob = np.array([increasing_prob])
        increasing_text = np.array([increasing_text])

    else:
        #here we have to deal with the rolling issue
        #find length of the time series
        time_series_length = len(confidence.get(min(confidence.keys())))

        rolling_confidence_dictionary = {}
        #build a new dictionary to iterate through
        increasing_prob = []
        increasing_text = []
        for cell_i in range(0,time_series_length):
            incr_prob,incr_text = probability_of_exceeding_target_slope({key_i:confidence.get(key_i)[cell_i] for key_i in confidence.keys()},
                                                                                    confidence_increasing_bins,
                                                                                    confidence_increasing_names,
                                                                                    target=target)
            increasing_prob.append(incr_prob)
            increasing_text.append(incr_text)
        increasing_prob = np.array(increasing_prob)
        increasing_text = np.array(increasing_text)

    return increasing_prob, increasing_text




###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
def trend_rolling(Y:              np.ndarray,
                  t:              np.ndarray,
                  t_unit:         str = '',
                  Y_unit:         str = '',
                  window:         int = 12,
                  alphas:         list = [0.05] + [x/20 for x in range(1,20)],
                  timestep:       float = 60*60*24,
                  timestep_unit:  str = 'day',
                  include_data:   bool = True,
                  legend_loc:     int = 2,
                  shade_area:     bool = False,
                  xaxis_rotation: float = 270
                  ):
    '''
    Function to calculate and plot trends using Rolling Ordinary Least Squares (R-OLS).

    Parameters
    ----------
    Y : np.ndarray
        DESCRIPTION: Input array of the original data.
    t : np.ndarray
        DESCRIPTION: Array of input times.
    t_unit : str, optional
        DESCRIPTION: Time unit. Not required if time is a datetime. The default is ''.
    Y_unit : str, optional
        DESCRIPTION. Data unit. The default is ''.
    window : int, optional
        DESCRIPTION. Length of the rolling window. Must be strictly larger than the number of variables in the model. The default is 12.
    alphas : list, optional
        DESCRIPTION. A list of significance levels for the confidence interval. For example, alpha = [.05] returns a 95% confidence interval. The default is [0.05] + [x/20 for x in range(1,20)].
    timestep : float, optional
        DESCRIPTION. Numeric timestep size in timestep_unit or seconds if the time arrayis a date. The default is 60*60*24.
    timestep_unit : str, optional
        DESCRIPTION. Timestep unit (e.g. seconds, days, years). The default is 'day'.
    include_data : bool, optional
        DESCRIPTION. Whether to include the original time-series in the plot or not. The default is True.
    legend_loc : int, optional
        DESCRIPTION: Location of the legend. The default is 2.
    shade_area : bool, optional
        DESCRIPTION: Whether to shade below the trend or not. The default is False.
    xaxis_rotation : float, optional
        DESCRIPTION: Angle (degrees) to control of the x-axis ticks. The default is 270.

    Returns
    -------
    fig : matplotlib.figure
        DESCRIPTION: Trend figure
    slopes : np.array
        DESCRIPTION: Array of slopes from the Rolling OLS
    increasing_prob, np.ndarray
        DESCRIPTION: Probability of exceeding the target slope.
    increasing_text : np.ndarray
        DESCRIPTION: Text classification of the probability of exceeding the target slope.
    confidence : dict
        DESCRIPTION: dictionary/numeric value of slopes.

    '''

    t_raw_ = copy.deepcopy(t)
    if isinstance(t[-1],np.datetime64):
        #convert to datetime.datetime
        t = [x.astype('datetime64[s]').astype(datetime.datetime) for x in t]

    if isinstance(t[-1],datetime.date):
        t = [datetime.datetime.combine(d, datetime.datetime.min.time()) for d in t]

    if isinstance(t[0],np.ndarray):
        t = np.array([dt[0].astype(datetime.datetime) for dt in t])

    if type(t[-1]) in [datetime.datetime]:
        t_ = np.array([dt.timestamp() for dt in t])
        t_ -= t_[0]
        #divide by timestep
        if timestep is not None:
            t_ = t_/timestep
            if timestep_unit is None:
                timestep_unit = str(timestep) + ' seconds'
        else:
            timestep_unit = 'second'
        t_ = sm.add_constant(t_, prepend=True) # add constant as the first column
    else:
        t_ = sm.add_constant(t, prepend=True) # add constant as the first column
        timestep_unit = t_unit


    model = RollingOLS(Y,t_,window=window)
    results = model.fit()

    #get Rsq
    rsq = results.rsquared

    #get mse
    mse = results.mse_resid

    #get predictions
    slopes = results.params[:,1]
    intercepts = results.params[:,0]

    #generate confidence bands for slope
    confidence_upper = {1 - ((a_i)/2 + (1-a_i)): results.conf_int(alpha=a_i)[:,1][:,1] for a_i in alphas}
    alphas = alphas[::-1]
    confidence_lower = {1 - a_i/2: results.conf_int(alpha=a_i)[:,0][:,1] for a_i in alphas}

    confidence = {}
    confidence.update(confidence_upper)
    confidence.update({0.5:slopes})
    confidence.update(confidence_lower)

    #classify the confidence
    increasing_prob, increasing_text  =  probabiliy_increasing_trend(confidence, rolling = True)

    #plot
    fig = plot_trend(Y,t_raw_,slopes,increasing_prob, increasing_text,include_data = include_data, legend_loc = legend_loc, shade_area=shade_area, xaxis_rotation = xaxis_rotation, t_unit = t_unit, Y_unit = Y_unit, timestep_unit = timestep_unit)

    return fig, slopes, increasing_prob, increasing_text, confidence

###############################################################################
###############################################################################
###############################################################################
def trend_linear(Y:              np.ndarray,
                 t:              np.ndarray,
                 t_unit:         str = '',
                 Y_unit:         str = '',
                 alphas:          list = [0.05] + [x/20 for x in range(1,20)],
                 timestep:       float|None = None,
                 timestep_unit:  str|None = None,
                 include_data:   bool = True,
                 legend_loc:     int = 2,
                 shade_area:     bool = False,
                 xaxis_rotation: float = 270
                 ):
    '''
    Function to calculate and plot trends using Ordinary Least Squares (OLS).

    Parameters
    ----------
    Y : np.ndarray
        DESCRIPTION: Input array of the original data.
    t : np.ndarray
        DESCRIPTION: Array of input times.
    t_unit : str, optional
        DESCRIPTION: Time unit. Not required if time is a datetime. The default is ''.
    Y_unit : str, optional
        DESCRIPTION. Data unit. The default is ''.
    alphas : list, optional
        DESCRIPTION. A list of significance levels for the confidence interval. For example, alpha = [.05] returns a 95% confidence interval. The default is [0.05] + [x/20 for x in range(1,20)].
    timestep : float, optional
        DESCRIPTION. Numeric timestep size in either timestep_unit or seconds if the time arrayis a date. The default is 60*60*24.
    timestep_unit : str, optional
        DESCRIPTION. Timestep unit (e.g. seconds, days, years). The default is 'day'.
    include_data : bool, optional
        DESCRIPTION. Whether to include the original time-series in the plot or not. The default is True.
    legend_loc : int, optional
        DESCRIPTION: Location of the legend. The default is 2.
    shade_area : bool, optional
        DESCRIPTION: Whether to shade below the trend or not. The default is False.
    xaxis_rotation : float, optional
        DESCRIPTION: Angle (degrees) to control of the x-axis ticks. The default is 270.

    Returns
    -------
    fig : matplotlib.figure
        DESCRIPTION: Trend figure
    slopes : float
        DESCRIPTION: Array of slopes from the Rolling OLS
    increasing_prob, float
        DESCRIPTION: Probability of exceeding the target slope.
    increasing_text : str
        DESCRIPTION: Text classification of the probability of exceeding the target slope.
    confidence : dict
        DESCRIPTION: dictionary/numeric value of slopes.

    '''
    t_raw_ = copy.deepcopy(t)
    if isinstance(t[-1],np.datetime64):
        #convert to datetime.datetime
        t = [x.astype('datetime64[s]').astype(datetime.datetime) for x in t]

    if isinstance(t[-1],datetime.date):
        t = [datetime.datetime.combine(d, datetime.datetime.min.time()) for d in t]

    if isinstance(t[0],np.ndarray):
        t = np.array([dt[0].astype(datetime.datetime) for dt in t])

    #
    if isinstance(t[-1],datetime.datetime):
        t_ = np.array([dt.timestamp() for dt in t])
        t_ -= t_[0]

        #divide by timestep
        if timestep is not None:
            t_ = t_/timestep
            if timestep_unit is None:
                timestep_unit = str(timestep) + ' seconds'

        else:
            timestep_unit = 'second'
        t_ = sm.add_constant(t_, prepend=True) # add constant as the first column
    else: #here we ASSUME a numerical array
        t_ = sm.add_constant(t, prepend=True) # add constant as the first column
        timestep_unit = t_unit

    model = sm.OLS(Y,t_)
    results = model.fit()

    #get Rsq
    rsq = results.rsquared
    #get mse
    mse = results.mse_resid

    slope = results.params[1]
    intercept = results.params[0]
    # print(slope)
    # print(intercept)
    #generate confidence bands for slope
    confidence_upper = {1 - ((a_i)/2 + (1-a_i)): results.conf_int(alpha=a_i)[1][1] for a_i in alphas}
    alphas = alphas[::-1]
    confidence_lower = {1 - (a_i/2): results.conf_int(alpha=a_i)[1][0] for a_i in alphas}

    confidence = {}
    confidence.update(confidence_upper)
    confidence.update({0.5:slope})
    confidence.update(confidence_lower)

    #classify the confidence
    increasing_prob, increasing_text  =  probabiliy_increasing_trend(confidence, rolling = False)


    #plot
    fig = plot_linear_trend(Y,t_raw_,timestep,slope,intercept,increasing_prob, np.repeat(increasing_text, len(t)), include_data = include_data, legend_loc = legend_loc, shade_area=shade_area, xaxis_rotation = xaxis_rotation,t_unit = t_unit, Y_unit = Y_unit, timestep_unit = timestep_unit)

    return fig, slope, increasing_prob, increasing_text, confidence
