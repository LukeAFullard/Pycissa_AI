import numpy as np
min_width = 720
min_height = 570
def normalise_matrix_array(my_array: np.ndarray) -> np.ndarray:
    '''
    Function to normalise an array the the sum of a row

    Parameters
    ----------
    my_array : np.ndarray
        DESCRIPTION: Inout array

    Returns
    -------
    new_array : np.ndarray
        DESCRIPTION: normalised array

    '''
    new_array = np.zeros(np.shape(my_array))
    for row_i in range(len(my_array[0,:])):
        new_array[:,row_i]  =  my_array[:,row_i] / sum(my_array[:,row_i])
    return new_array


def calculate_instantaneous_properties(x:                  np.ndarray,
                                       sampling_frequency: int = 1,
                                       N:                  int = None,
                                       axis:               int = -1):
    '''
    Function to calculate instantaneous amplitude, power, phase_angle, and frequency.
    Achieved by calculating the analytic function of x using the Hilbert transform.

    Parameters
    ----------
    x : numpy array
        DESCRIPTION: Input signal.
    sampling_frequency : numeric, optional
        DESCRIPTION: The default is 1. Sampling frequency of the signal (for calculating the time derivative needed for instantaneous frequency)
    N : int, optional
        DESCRIPTION: The default is None. Number of Fourier components.
    axis : int, optional
        DESCRIPTION: The default is -1. Axis along which to do the transformation.

    Returns
    -------
    amplitude : numpy array
        DESCRIPTION: The instantaneous amplitude of the input signal.
    power : numpy array
        DESCRIPTION: The instantaneous power of the input signal.
    phase_angle : numpy array
        DESCRIPTION: The instantaneous phase angle of the input signal.
    frequency : numpy array
        DESCRIPTION: The instantaneous frequency of the input signal.

    '''
    ##############################
    def calc_amplitude_power(f: np.ndarray):
        '''
        Function to calculate the instantaneous ampltude and power.
        '''
        amplitude = np.abs(f)
        power = amplitude**2
        return amplitude,power
    ##############################
    def calc_phase(f:   np.ndarray,
                   deg: bool = False):
        '''
        Function to calculate the instantaneous phase.
        '''
        phase_angle = np.angle(f, deg=deg)
        return phase_angle
    ##############################
    def calc_frequency(f:  np.ndarray,
                       dt: float):
        '''
        Function to calculate the instantaneous frequency.
        '''
        frequency = np.gradient(f,dt)
        return frequency
    ##############################

    from scipy.signal import hilbert
    #calculate the analytic function of x
    xa = hilbert(x,N=N, axis=axis)

    amplitude,power = calc_amplitude_power(xa)
    phase_angle = calc_phase(xa)
    frequency = calc_frequency(phase_angle,1/sampling_frequency)
    return amplitude,power,phase_angle,frequency


###############################################################################
def generate_f_t_matrix(x:                  np.ndarray,
                        sampling_frequency: float =1):
    '''
    Function which iterates through the frequencies obtained from CiSSA and combines these with instantaneous properties to generate heights for our frequency-time plots.

    Parameters
    ----------
    x : dict
        DESCRIPTION: Dictionary of frequency (keys) and cell reference (values).
    sampling_frequency : numeric, optional
        DESCRIPTION: The default is 1. Sampling frequency of the signal (for calculating the time derivative needed for instantaneous frequency)

    Returns
    -------
    freq_list : list
        DESCRIPTION: List of frequencies of the signals obtained via CiSSA
    amplitude_matrix : numpy array
        DESCRIPTION: Matrix of instantaneous amplitudes at the given frequencies obtained via CiSSA.
    power_matrix : numpy array
        DESCRIPTION: Matrix of instantaneous power at the given frequencies obtained via CiSSA.
    phase_matrix : numpy array
        DESCRIPTION: Matrix of instantaneous phase at the given frequencies obtained via CiSSA.
    frequency_matrix : numpy array
        DESCRIPTION: Matrix of instantaneous frequencies at the given frequencies obtained via CiSSA.

    '''
    freq_list = []
    amplitude_matrix, power_matrix, phase_matrix, frequency_matrix = None,None,None,None

    for freq_i,signal_i in x.items():
        freq_list.append(freq_i)
        signal_j =  np.ndarray.flatten(signal_i)

        amplitude,power,phase_angle,frequency = None,None,None,None
        amplitude,power,phase_angle,frequency = calculate_instantaneous_properties(signal_j, sampling_frequency = sampling_frequency)

        #amplitude matrix
        if amplitude_matrix is not None: amplitude_matrix = np.append(amplitude_matrix,np.array([amplitude]),axis = 0)
        else: amplitude_matrix = np.array([amplitude])

        #power matrix
        if power_matrix is not None: power_matrix = np.append(power_matrix,np.array([power]),axis = 0)
        else: power_matrix = np.array([power])

        #phase matrix
        if phase_matrix is not None: phase_matrix = np.append(phase_matrix,np.array([phase_angle]),axis = 0)
        else: phase_matrix = np.array([phase_angle])

        #frequency matrix
        if frequency_matrix is not None: frequency_matrix = np.append(frequency_matrix,np.array([frequency]),axis = 0)
        else: frequency_matrix = np.array([frequency])

    return freq_list, amplitude_matrix, power_matrix, phase_matrix, frequency_matrix




###############################################################################
def plot_frequency_time(t:       np.ndarray,
                        f:       np.ndarray,
                        z:       np.ndarray,
                        xsize:   int=6,
                        ysize:   int=6,
                        logplot: bool = False,
                        t_unit:  str = '',
                        Y_unit:  str = ''):
    '''
    A function to generate a frequency-time plot.

    Parameters
    ----------
    t : np.ndarray
        DESCRIPTION: Array of times/dates.
    f : np.ndarray
        DESCRIPTION: Array of frequencies.
    z : np.ndarray
        DESCRIPTION: Array defining "heights" at the given frequency/time for plotting.
    xsize : int, optional
        DESCRIPTION: The default is 6. Horizontal size of generated figure.
    ysize : int, optional
        DESCRIPTION: The default is 6. Vertical size of generated figure.
    logplot : bool, optional
        DESCRIPTION: Flat showing whether the plot should be on a log y-scale or not. The default is False.
    t_unit : str, optional
        DESCRIPTION: The default is ''. Name of the t-unit.
    Y_unit : str, optional
        DESCRIPTION: The default is ''. Name of the y-unit.

    Returns
    -------
    fig : matplotlib.figure
        DESCRIPTION: Figure of the frequency time plot
    ax : matplotlib.axes
        DESCRIPTION: Axes of the frequency time plot

    '''

    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    plt.style.use('ggplot')  # or 'seaborn', 'bmh', 'classic', etc.
    X, Y = np.meshgrid(t, f)
    Z = z

    # plot
    fig, ax = plt.subplots(figsize=(xsize, ysize))

    if logplot:
        myplot = ax.pcolormesh(X, Y, Z,norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()))
        clb = fig.colorbar(myplot)
    else:
        myplot = ax.pcolormesh(X, Y, Z)
        clb = fig.colorbar(myplot)
    ax.set_xlabel(t_unit)
    ax.set_ylabel("Frequency")
    clb.ax.set_title(Y_unit)

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
def plot_period_time(t:           np.ndarray,
                     periods:     np.ndarray,
                     z:           np.ndarray,
                     xsize:       int = 6,
                     ysize:       int = 6,
                     logplot:     bool = False,
                     t_unit:      str = '',
                     Y_unit:      str = '',
                     period_name: str = ''):
    '''
    A function to generate a period-time plot.

    Parameters
    ----------
    t : np.ndarray
        DESCRIPTION: Array of times/dates.
    periods : np.ndarray
        DESCRIPTION: Array of periods/cycles from the CiSSA analysis.
    z : np.ndarray
        DESCRIPTION: Array defining "heights" at the given period/time for plotting.
    xsize : int, optional
        DESCRIPTION: The default is 6. Horizontal size of generated figure.
    ysize : int, optional
        DESCRIPTION: The default is 6. Vertical size of generated figure.
    logplot : bool, optional
        DESCRIPTION: Flat showing whether the plot should be on a log y-scale or not. The default is False.
    t_unit : str, optional
        DESCRIPTION: The default is ''. Name of the t-unit.
    Y_unit : str, optional
        DESCRIPTION: The default is ''. Name of the y-unit.
    period_name : str, optional
        DESCRIPTION: The default is ''. Name of the period.

    Returns
    -------
    fig : matplotlib.figure
        DESCRIPTION: Figure of the frequency time plot
    ax : matplotlib.axes
        DESCRIPTION: Axes of the frequency time plot

    '''
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    plt.style.use('ggplot')  # or 'seaborn', 'bmh', 'classic', etc.
    X, Y = np.meshgrid(t, periods)
    Z = z

    # plot
    fig, ax = plt.subplots(figsize=(xsize, ysize))

    if logplot:
        myplot = ax.pcolormesh(X, Y, Z,norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()))
        clb = fig.colorbar(myplot)
    else:
        myplot = ax.pcolormesh(X, Y, Z)
        clb = fig.colorbar(myplot)

    ax.set_xlabel(t_unit)
    ax.set_ylabel(f"Period ({period_name})")
    clb.ax.set_title(Y_unit)

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
###############################################################################
def _run_frequency_time_analysis(Z:                  np.ndarray,
                                psd:                np.ndarray,
                                t:                  np.ndarray,
                                L:                  int,
                                data_per_period:    int,
                                period_name:        str = '',
                                t_unit:             str = '',
                                plot_frequency:     bool = True,
                                plot_period:        bool = False,
                                logplot_frequency:  bool = False,
                                logplot_period:     bool = False,
                                normalise_plots:    bool = False,
                                height_variable:    str = 'power',
                                height_unit:        str = '',
                                ):
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
    from pycissa.postprocessing.grouping.grouping_functions import group, generate_grouping
    sampling_frequency = 1/data_per_period
    #Generate grouping
    I = generate_grouping(psd,L)

    #Group frequency components
    rc, sh, kg, psd_sh = group(Z,psd,I)

    #generate matrix
    freq_list, amplitude_matrix, power_matrix, phase_matrix, frequency_matrix = generate_f_t_matrix(rc,sampling_frequency=sampling_frequency)
    period_list = [1/(x*data_per_period) for x in freq_list]
    if height_variable == 'power':
        if normalise_plots:
            z = normalise_matrix_array(power_matrix)
            height_unit = 'Normalised ' + height_variable + f'({height_unit}^2)'
        else:
            z =  power_matrix
            height_unit = height_variable  + f'({height_unit}^2)'
    elif height_variable == 'amplitude':
        if normalise_plots:
            z = normalise_matrix_array(amplitude_matrix)
            height_unit = 'Normalised ' + height_variable + f'({height_unit})'
        else:
            z = amplitude_matrix
            height_unit = height_variable  + f'({height_unit})'
    elif height_variable == 'phase':
        if normalise_plots:
            z = normalise_matrix_array(phase_matrix)
            height_unit = 'Normalised ' + height_variable
        else:
            z = phase_matrix
            height_unit = height_variable
    else:
        raise ValueError(f'variable height_variable = {height_variable} but should be one of power, amplitude, or phase')

    fig_f,ax_f,fig_p,ax_p = None,None,None,None
    if plot_frequency:
        fig_f,ax_f = plot_frequency_time(t,freq_list,z,logplot = logplot_frequency,t_unit = t_unit, Y_unit = height_unit)
    if plot_period:
        fig_p,ax_p = plot_period_time(t,period_list,z,logplot = logplot_period,t_unit = t_unit, Y_unit = height_unit,period_name=period_name)

    return freq_list, period_list, amplitude_matrix, power_matrix, phase_matrix, frequency_matrix, fig_f,fig_p
