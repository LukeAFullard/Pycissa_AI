import numpy as np
import statsmodels.api as sm
from pycissa.processing.mcissa.mcissa import MCissa
import matplotlib.pyplot as plt
import os

output_dir = "examples/mcissa_benchmarks"
os.makedirs(output_dir, exist_ok=True)

df = sm.datasets.elnino.load_pandas().data
X = df.iloc[:, 1:].values
t_years = df['YEAR'].values

mcissa = MCissa(t=t_years, x=X)
mcissa.fit(L=16)
mcissa.post_run_frequency_time_analysis(data_per_period=1)
mcissa.auto_cissa(L=16, plot_result=False, verbose=False)

# The plot methods return the MCissa object itself, not the figure.
# We need to use matplotlib to save the current figure.
mcissa.plot_original_time_series()
plt.savefig(os.path.join(output_dir, "elnino_time_series.png"), bbox_inches='tight')
plt.close()

mcissa.plot_components(num_components=3)
plt.savefig(os.path.join(output_dir, "elnino_components.png"), bbox_inches='tight')
plt.close()

print("Plots saved successfully!")
