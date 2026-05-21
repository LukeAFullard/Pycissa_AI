import numpy as np
import matplotlib.pyplot as plt
from pycissa.processing.mcissa.mcissa import MCissa
import os

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
T = 300
t = np.arange(T)

# Main channel: trend + slow cycle + noise
trend = 0.05 * t
slow_cycle = 2 * np.sin(2 * np.pi * t / 50)
main_noise = np.random.normal(0, 0.5, T)
x_main = trend + slow_cycle + main_noise

# Reference channel 1: Contains the same slow cycle + noise
# PLUS a fast cycle that is highly significant in the reference, but totally absent in the main signal
ref_cycle = 1.5 * np.sin(2 * np.pi * t / 50 + np.pi/4)
fast_cycle_only_in_ref = 3.0 * np.sin(2 * np.pi * t / 10)
x_ref1 = ref_cycle + fast_cycle_only_in_ref + np.random.normal(0, 0.5, T)

# Combine into a 2D array: (T, M)
x = np.column_stack([x_main, x_ref1])

print("Fitting MCissa...")
mcissa = MCissa(t, x)
# Fit with window length L
L = 60
mcissa.fit(L)

# 1. Standard Post-processing Analysis (Before BSS)
print("Grouping components...")
mcissa.post_group_components(grouping_type='smallest_proportion', eigenvalue_proportion=0.9, plot_result=False)

print("Running Frequency-Time Analysis on Main Channel (Before BSS)...")
mcissa.post_run_frequency_time_analysis(
    data_per_period=10,
    channel_index=0,
    logplot_frequency=False,
    period_name='samples',
    t_unit='time',
    height_variable='amplitude'
)
fig_ft_before = mcissa.figures['mcissa']['figure_frequency_time_channel_0']
fig_ft_before.savefig(os.path.join(os.path.dirname(__file__), 'mcissa_freq_time_before.png'), bbox_inches='tight')

print("Running Trend Analysis on Main Channel (Before BSS)...")
mcissa.post_analyse_trend(
    channel_index=0,
    trend_type='linear',
    t_unit='time',
    data_unit='value'
)
fig_trend_before = mcissa.figures['mcissa']['figure_trend_channel_0']
fig_trend_before.savefig(os.path.join(os.path.dirname(__file__), 'mcissa_trend_before.png'), bbox_inches='tight')

# 2. Blind Source Separation
print("Running Blind Source Separation...")
# Identify components in the main signal that are shared with the reference signal
mcissa.auto_blind_source_separation(main_index=0, K_surrogates=1, alpha=0.05)

# 3. Post-processing Analysis (After BSS)
print("Running Frequency-Time Analysis on Main Channel (After BSS)...")
# By using use_cleaned=True, we analyze the main channel minus the reference influences (which removes the slow cycle)
mcissa.post_run_frequency_time_analysis(
    data_per_period=10,
    use_cleaned=True,
    logplot_frequency=False,
    period_name='samples',
    t_unit='time',
    height_variable='amplitude'
)
fig_ft_after = mcissa.figures['mcissa']['figure_frequency_time_channel_0_cleaned']
fig_ft_after.savefig(os.path.join(os.path.dirname(__file__), 'mcissa_freq_time_after.png'), bbox_inches='tight')

print("Running Trend Analysis on Main Channel (After BSS)...")
mcissa.post_analyse_trend(
    use_cleaned=True,
    trend_type='linear',
    t_unit='time',
    data_unit='value'
)
fig_trend_after = mcissa.figures['mcissa']['figure_trend_channel_0_cleaned']
fig_trend_after.savefig(os.path.join(os.path.dirname(__file__), 'mcissa_trend_after.png'), bbox_inches='tight')

print("Generating plots comparing signals...")
# Generate a summary plot of original vs BSS cleaned
plt.figure(figsize=(10, 6))
plt.plot(t, x_main, label='Original Main Signal (Trend + Slow Cycle + Noise)', color='black', alpha=0.7)
plt.plot(t, mcissa.x_cleaned, label='BSS Cleaned Main Signal (Slow Cycle removed)', color='blue', linewidth=2)
plt.plot(t, trend, label='Underlying Ground Truth Trend', color='red', linestyle='dashed', linewidth=2)
plt.legend()
plt.title('Main Signal Before and After Blind Source Separation\n(Note: Fast cycle in reference does not disturb main signal)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.savefig(os.path.join(os.path.dirname(__file__), 'mcissa_bss_summary.png'), bbox_inches='tight')
plt.close()

print("All examples generated successfully.")
