import numpy as np
import pytest
from pycissa.processing.mcissa.mcissa import MCissa

def test_mcissa_new_methods():
    t = np.arange(100)
    x = np.random.rand(100, 2)
    mcissa = MCissa(t, x)
    mcissa.fit(10)

    mcissa.post_group_components(grouping_type='smallest_n', plot_result=False)

    # Test frequency time analysis on channel 0
    mcissa.post_run_frequency_time_analysis(data_per_period=10, channel_index=0, logplot_frequency=False)
    assert 'figure_frequency_time_channel_0' in mcissa.figures['mcissa']

    # Test frequency time analysis on channel 1
    mcissa.post_run_frequency_time_analysis(data_per_period=10, channel_index=1, logplot_frequency=False)
    assert 'figure_frequency_time_channel_1' in mcissa.figures['mcissa']

    # Test trend
    mcissa.post_analyse_trend(channel_index=0)
    assert 'figure_trend_channel_0' in mcissa.figures['mcissa']

    # Test BSS
    mcissa.auto_blind_source_separation(main_index=0)
    assert hasattr(mcissa, 'x_cleaned_components')
    assert mcissa.x_cleaned_components.shape == (100, mcissa.Z_stacked.shape[2])

    # Test after BSS
    mcissa.post_run_frequency_time_analysis(data_per_period=10, use_cleaned=True, logplot_frequency=False)
    assert 'figure_frequency_time_channel_0_cleaned' in mcissa.figures['mcissa']

    mcissa.post_analyse_trend(use_cleaned=True)
    assert 'figure_trend_channel_0_cleaned' in mcissa.figures['mcissa']

def test_mcissa_post_periodogram():
    t = np.arange(200)
    # Add a trend, some periodic signal and noise to both channels
    trend = 0.05 * t
    periodic = np.sin(2 * np.pi * t / 20)
    noise = np.random.randn(200) * 0.1
    x1 = trend + periodic + noise
    x2 = 0.5 * trend + 1.5 * periodic + noise
    x = np.column_stack((x1, x2))

    mcissa = MCissa(t, x)
    mcissa.fit(L=20)

    # We must run auto_detrend or post_group_components first
    mcissa.post_group_components(grouping_type='smallest_n', number_of_groups_to_drop=15, plot_result=False)

    # Run periodogram analysis on channel 0
    mcissa.post_periodogram_analysis(channel_index=0, significant_components=[], monte_carlo_significant_components=False)

    # Verify figures and results were created
    assert 'figure_periodogram_linear_channel_0' in mcissa.figures['mcissa']
    assert '_channel_0' in mcissa.results['mcissa']['fractal scaling results']
    assert 'full Hurst exponent' in mcissa.results['mcissa']['fractal scaling results']['_channel_0']

    # Run periodogram analysis on channel 1
    mcissa.post_periodogram_analysis(channel_index=1, significant_components=[], monte_carlo_significant_components=False)
    assert 'figure_periodogram_linear_channel_1' in mcissa.figures['mcissa']
    assert '_channel_1' in mcissa.results['mcissa']['fractal scaling results']

    # Now test with use_cleaned
    mcissa.auto_blind_source_separation(main_index=0, K_surrogates=10) # dummy K to be fast
    mcissa.post_periodogram_analysis(use_cleaned=True, significant_components=[], monte_carlo_significant_components=False)
    assert 'figure_periodogram_linear_cleaned' in mcissa.figures['mcissa']
    assert '_cleaned' in mcissa.results['mcissa']['fractal scaling results']

if __name__ == "__main__":
    test_mcissa_new_methods()
    test_mcissa_post_periodogram()
    print("All tests passed.")
