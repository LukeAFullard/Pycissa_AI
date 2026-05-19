import numpy as np
from pycissa.processing.mcissa.mcissa import MCissa

np.random.seed(42)
T = 300
L = 36
M = 3

t = np.arange(1, T + 1)

# Trend
trend1 = 0.05 * t
trend2 = -0.02 * t
trend3 = 0.01 * t

# Low-frequency wave
low_freq1 = 3.0 * np.sin(2 * np.pi * t / 24)
low_freq2 = 2.5 * np.cos(2 * np.pi * t / 24)
low_freq3 = 1.0 * np.sin(2 * np.pi * t / 24 + np.pi / 4)

# High-frequency wave
high_freq1 = 1.5 * np.sin(2 * np.pi * t / 6)
high_freq2 = 1.5 * np.sin(2 * np.pi * t / 6 - np.pi / 3)
high_freq3 = 2.0 * np.cos(2 * np.pi * t / 6)

noise1 = np.random.normal(0, 0.1, T)
noise2 = np.random.normal(0, 0.1, T)
noise3 = np.random.normal(0, 0.1, T)

x1 = trend1 + low_freq1 + high_freq1 + noise1
x2 = trend2 + low_freq2 + high_freq2 + noise2
x3 = trend3 + low_freq3 + high_freq3 + noise3

X = np.column_stack((x1, x2, x3))

mcissa = MCissa(t, X)
mcissa.fit(L=L, extension_type='NoExt')

fig = mcissa.plot_components(num_components=3, variable_names=['Var 1', 'Var 2', 'Var 3'], component_names=['Trend', 'Low Freq', 'High Freq'])

fig.savefig('mcissa_example_plot.png')
print("Saved example plot to mcissa_example_plot.png")
