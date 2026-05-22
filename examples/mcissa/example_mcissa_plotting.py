import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pycissa.processing.mcissa.mcissa import MCissa

# Set random seed for reproducibility
np.random.seed(42)

# Generate 3 years of daily data
dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(3 * 365)]
T = len(dates)
t_array = np.arange(T)

# Channel 1: Trend + Yearly Seasonality + Noise
trend1 = 0.01 * t_array
seasonality1 = 5 * np.sin(2 * np.pi * t_array / 365)
noise1 = np.random.normal(0, 1, T)
x1 = trend1 + seasonality1 + noise1

# Channel 2: Stronger Trend + Shifted Seasonality + Noise
trend2 = 0.02 * t_array
seasonality2 = 3 * np.sin(2 * np.pi * t_array / 365 + np.pi/4)
noise2 = np.random.normal(0, 2, T)
x2 = trend2 + seasonality2 + noise2

x = np.column_stack([x1, x2])

print("Initializing MCissa...")
mcissa = MCissa(dates, x)

print("Plotting original multivariate time series...")
mcissa.plot_original_time_series()
fig_orig = mcissa.figures['mcissa']['figure_original_time_series']
fig_orig.savefig("mcissa_original_time_series.png")

print("Fitting MCissa...")
mcissa.fit(L=100)

print("Grouping components...")
mcissa.post_group_components(grouping_type='smallest_proportion', eigenvalue_proportion=0.9, plot_result=False)

print("Plotting monthly seasonal boxplots...")
mcissa.plot_seasonal_boxplots(plot_type='monthly')
fig_monthly = mcissa.figures['mcissa']['figure_monthly_seasonal_box']
fig_monthly.savefig("mcissa_monthly_boxplots.png")

print("Plotting yearly seasonal boxplots...")
mcissa.plot_seasonal_boxplots(plot_type='yearly')
fig_yearly = mcissa.figures['mcissa']['figure_yearly_seasonal_box']
fig_yearly.savefig("mcissa_yearly_boxplots.png")

print("Done. Saved plots to current directory.")
