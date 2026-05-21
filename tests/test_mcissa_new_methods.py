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

if __name__ == "__main__":
    test_mcissa_new_methods()
    print("All tests passed.")
