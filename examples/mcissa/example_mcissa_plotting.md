# MCissa Plotting Examples

This example demonstrates how to use the multivariate plotting methods recently added to `MCissa`, notably `plot_original_time_series` and `plot_seasonal_boxplots`.

## Generating the Synthetic Data

We generate a synthetic two-channel dataset over three years.

```python
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pycissa.processing.mcissa.mcissa import MCissa

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

# Initialize MCissa
mcissa = MCissa(dates, x)
```

## Plotting the Original Time Series

You can easily visualize the multi-channel original time series:

```python
# Plots M channels vertically stacked
mcissa.plot_original_time_series()
```

## Plotting Seasonal Boxplots

You can plot seasonal variations split by month or year across all channels.
First, we fit and group the components so we can isolate the periodic behavior without the trend if we wish.

```python
mcissa.fit(L=100)
mcissa.post_group_components(grouping_type='smallest_proportion', eigenvalue_proportion=0.9, plot_result=False)

# Plot seasonal boxplots split by month
mcissa.plot_seasonal_boxplots(plot_type='monthly')

# Plot seasonal boxplots split by year
mcissa.plot_seasonal_boxplots(plot_type='yearly')
```
