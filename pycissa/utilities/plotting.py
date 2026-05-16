import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import copy
min_width = 720
min_height = 570
#########################################################################
#########################################################################
#########################################################################

def plot_time_series(t: np.ndarray,
                            x: np.ndarray,):
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(figsize=(10, 8), sharex=True)

    # Plot for the top subplot
    axs.plot(t, x, 'k', label='original time-series')

    # Set labels and title for the top subplot
    axs.set_ylabel('value')
    axs.legend(loc='upper left')
    axs.set_title('Time Series')

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
#########################################################################
#########################################################################
#########################################################################
def plot_grouped_components(t: np.ndarray,
                            x: np.ndarray,
                            x_trend: np.ndarray,
                            x_periodic: np.ndarray,
                            x_noise: np.ndarray):
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot for the top subplot
    axs[0].plot(t, x, 'k', label='original time-series')
    axs[0].plot(t, x_trend, 'r', label='trend')
    axs[0].plot(t, x_periodic, 'g', label='periodic')
    axs[0].plot(t, x_trend + x_periodic, 'b', label='trend + periodic')

    # Set labels and title for the top subplot
    axs[0].set_ylabel('value')
    axs[0].legend(loc='upper left')
    axs[0].set_title('Time Series Components')

    # Plot for the bottom subplot
    axs[1].plot(t, x_noise, 'm', label='noise')
    axs[1].plot(t, x_periodic, 'g', label='periodic')

    # Set labels and title for the bottom subplot
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('value')
    axs[1].legend(loc='upper left')
    axs[1].set_title('Noise and Periodic Components')

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
#########################################################################
#########################################################################
#########################################################################
def plot_noise_residual(x: np.ndarray, x_noise: np.ndarray):
    # Create a figure and a set of subplots with 1 row and 2 columns
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

    # Plot for the first subplot (scatter plot of residuals)
    axs[0].scatter(x, x_noise, color='k')
    axs[0].set_ylabel('Residual')
    axs[0].set_xlabel('Original time series value')
    axs[0].set_title('Noise/Residual Plot')
    axs[0].axhline(y=0, color='r', linestyle='--', linewidth=1.5)

    # Compute histogram data
    n_bins = 30
    counts, bins, patches = axs[1].hist(x_noise, bins=n_bins, color='b', edgecolor='black', alpha=0.7)

    # Set x-axis and y-axis limits for histogram
    axs[1].set_xlabel('Residual')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Histogram of Residuals')
    axs[1].set_ylim(0, np.max(counts) * 1.1)  # 10% more than the maximum count

    # Adjust layout to prevent overlap
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

#########################################################################
#########################################################################
#########################################################################
def plot_detrended_signal(t: np.ndarray,
                            x: np.ndarray,
                            x_trend: np.ndarray,
                            x_detrended: np.ndarray,
                            ):
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot for the top subplot
    axs[0].plot(t, x, 'k', label='original time-series')
    axs[0].plot(t, x_trend, 'r', label='trend')

    # Set labels and title for the top subplot
    axs[0].set_ylabel('value')
    axs[0].legend(loc='upper left')
    axs[0].set_title('Original time series with trend')

    # Plot for the bottom subplot
    axs[1].plot(t, x_detrended, 'g', label='detrended')

    # Set labels and title for the bottom subplot
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('value')
    axs[1].legend(loc='upper left')
    axs[1].set_title('Detrended signal')

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
#########################################################################
#########################################################################
#########################################################################
def plot_denoised_signal(t: np.ndarray,
                            x: np.ndarray,
                            x_denoised: np.ndarray,
                            ):
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot for the top subplot
    axs[0].plot(t, x, 'k', label='original time-series')

    # Set labels and title for the top subplot
    axs[0].set_ylabel('value')
    axs[0].legend(loc='upper left')
    axs[0].set_title('Original time series')

    # Plot for the bottom subplot
    axs[1].plot(t, x_denoised, 'g', label='denoised')

    # Set labels and title for the bottom subplot
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('value')
    axs[1].legend(loc='upper left')
    axs[1].set_title('Denoised time series')

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
#########################################################################
#########################################################################
#########################################################################
def seasonal_boxplots(t:          np.ndarray,
                      x:          np.ndarray,
                      split_date: datetime|None = None,
                      bar_width:  float = 0.25):
    '''
    Plotting function to create a seasonal boxplot split by months.
    Includes the option to split the data by a set date and will plot grouped boxplots with one group being the boxplot of the seasonal data before the set date, the other after.

    Parameters
    ----------
    t : np.ndarray
        DESCRIPTION.Array of input times.
    x : np.ndarray
        DESCRIPTION. Array of input data.
    split_date : datetime|None, optional
        DESCRIPTION. The default is None. A datetime object which splits the boxplots into groups, one before the set date, one after. If None, the data is not split.
    bar_width : float, optional
        DESCRIPTION. The default is 0.25. Width of each individual bar

    Returns
    -------
    fig : Figure
        DESCRIPTION. Boxplots

    '''
    bar_offset = 1.15*(bar_width/2)
    t_ = copy.deepcopy(t)
    t_ = [np.datetime64(dt, 's') for dt in t_]
    t_ = [dt.astype(datetime) for dt in t_]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    boxprops = dict(linewidth=1.5, color='darkblue')  # Box color
    whiskerprops = dict(linewidth=1.5, color='black')  # Whisker color
    capprops = dict(linewidth=1.5, color='black')      # Cap color
    medianprops = dict(linewidth=2, color='red')       # Median line color
    #get months
    months = ['Jan','Feb','Mar','Apr','May','Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    data = []
    if split_date is not None:
        if not type(split_date) in [datetime]:raise TypeError(f"Imput variable split_date should be a datetime object. Currently is if of type {type(split_date)}")

        #split data into two parts
        t_before = [y < split_date for y in t_]
        t0 = [y for y,w in zip(t_,t_before) if w == True]
        x0 = [y for y,w in zip(x,t_before) if w == True]
        t1 = [y for y,w in zip(t_,t_before) if w == False]
        x1 = [y for y,w in zip(x,t_before) if w == False]

        for i,month_j in enumerate(months):
            data.append(np.array([w for y,w in zip(t0,x0) if y.strftime('%b') == month_j]))

        # Create the boxplot
        bp0 = ax.boxplot(data, labels=months,
                   notch=False,                  # Add notches
                    patch_artist=True,           # Enable coloring of boxes
                    boxprops=boxprops,           # Set box color)
                    whiskerprops=whiskerprops,   # Set whisker color
                    capprops=capprops,           # Set cap color
                    medianprops=medianprops,
                    widths = 0.2)

        for i,whisker in enumerate(bp0['whiskers']):
            whisker.set_xdata(whisker.get_xdata()-bar_offset)
        for i,cap in enumerate(bp0['caps']):
            cap.set_xdata(cap.get_xdata()-bar_offset)
        for i,median in enumerate(bp0['medians']):
            median.set_xdata(median.get_xdata()-bar_offset)
        for i,flier in enumerate(bp0['fliers']):
            flier.set_xdata(flier.get_xdata()-bar_offset)

        for i,patch in enumerate(bp0['boxes']):
            patch.set_facecolor('lightblue')
            if i == 0:
                patch.set_label(f"Before {split_date.date().strftime('%d %b %Y')}")
            for vertex_i in patch.get_path().vertices:
                vertex_i[0] += -bar_offset
            patch.set_path(patch.get_path())
        ax.set_xticks([])

        data = []
        for i,month_j in enumerate(months):
            data.append(np.array([w for y,w in zip(t1,x1) if y.strftime('%b') == month_j]))


        bp1 = ax.boxplot(data, labels=months,
                   notch=False,                  # Add notches
                    patch_artist=True,           # Enable coloring of boxes
                    boxprops=boxprops,           # Set box color)
                    whiskerprops=whiskerprops,   # Set whisker color
                    capprops=capprops,           # Set cap color
                    medianprops=medianprops,
                    widths = bar_width)
        for i,whisker in enumerate(bp1['whiskers']):
            whisker.set_xdata(whisker.get_xdata()+bar_offset)
        for i,cap in enumerate(bp1['caps']):
            cap.set_xdata(cap.get_xdata()+bar_offset)
        for i,median in enumerate(bp1['medians']):
            median.set_xdata(median.get_xdata()+bar_offset)
        for i,flier in enumerate(bp1['fliers']):
            flier.set_xdata(flier.get_xdata()+bar_offset)
        for i,patch in enumerate(bp1['boxes']):
            patch.set_facecolor('lightgreen')
            if i == 0:
                patch.set_label(f"After {split_date.date().strftime('%d %b %Y')}")
            for vertex_i in patch.get_path().vertices:
                vertex_i[0] += bar_offset
            patch.set_path(patch.get_path())
        # ax.set_title('Boxplot of Two Groups')
        ax.set_ylabel('Value')
        ax.legend(loc='upper center')

    else:
        for month_j in months:
            data.append(np.array([w for y,w in zip(t_,x) if y.strftime('%b') == month_j]))

        # Create the boxplot
        bp = ax.boxplot(data, labels=months,
                   notch=False,                  # Add notches
                    patch_artist=True,           # Enable coloring of boxes
                    boxprops=boxprops,           # Set box color)
                    whiskerprops=whiskerprops,   # Set whisker color
                    capprops=capprops,           # Set cap color
                    medianprops=medianprops,
                    widths = bar_width)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        # Add title and labels
        ax.set_ylabel('Value')
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




def yearly_boxplots(t:          np.ndarray,
                      x:          np.ndarray,
                      bar_width:  float = 0.25):
    '''
    Plotting function to create a seasonal boxplot split by years.

    Parameters
    ----------
    t : np.ndarray
        DESCRIPTION.Array of input times.
    x : np.ndarray
        DESCRIPTION. Array of input data.
    bar_width : float, optional
        DESCRIPTION. The default is 0.25. Width of each individual bar

    Returns
    -------
    fig : Figure
        DESCRIPTION. Boxplots

    '''

    t_ = copy.deepcopy(t)
    t_ = [np.datetime64(dt, 's') for dt in t_]
    t_ = [dt.astype(datetime) for dt in t_]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    boxprops = dict(linewidth=1.5, color='darkblue')  # Box color
    whiskerprops = dict(linewidth=1.5, color='black')  # Whisker color
    capprops = dict(linewidth=1.5, color='black')      # Cap color
    medianprops = dict(linewidth=2, color='red')       # Median line color
    #get months
    years = [x.year for x in t_]
    years = sorted(list(set(years)))
    data = []
    for i,year_j in enumerate(years):
        data.append(np.array([w for y,w in zip(t_,x) if y.year == year_j]))

    # Create the boxplot
    bp = ax.boxplot(data, labels=years,
               notch=False,                  # Add notches
                patch_artist=True,           # Enable coloring of boxes
                boxprops=boxprops,           # Set box color)
                whiskerprops=whiskerprops,   # Set whisker color
                capprops=capprops,           # Set cap color
                medianprops=medianprops,
                widths = bar_width)



    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    # Add title and labels
    ax.set_ylabel('Value')
    ax.set_xticklabels(years,rotation=90, ha='right')  # Rotate labels
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


