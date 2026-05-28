import numpy as np
import statsmodels.api as sm
from pycissa.processing.mcissa.mcissa import MCissa
from pycissa.processing.cissa.cissa import Cissa
import matplotlib.pyplot as plt
import os

output_dir = "examples/mcissa_benchmarks"
os.makedirs(output_dir, exist_ok=True)

df = sm.datasets.elnino.load_pandas().data
X = df.iloc[:, 1:].values # 12 months as channels (multivariate)
t_years = df['YEAR'].values

print("--- 1. RUNNING UNIVARIATE CISSA (Annual Mean) ---")
# Calculate Annual Mean for the Univariate comparison
x_annual_mean = np.mean(X, axis=1)

cissa_annual = Cissa(t=t_years, x=x_annual_mean)
cissa_annual.fit(L=16)
cissa_annual.post_run_frequency_time_analysis(data_per_period=1)
cissa_annual.auto_cissa(L=16, plot_result=False, verbose=False)

cissa_annual.plot_original_time_series()
plt.title("Univariate El Niño (Annual Mean Sea Surface Temp)")
plt.xlabel("Year")
plt.ylabel("Temperature Anomaly")
plt.savefig(os.path.join(output_dir, "elnino_cissa_annual_time_series.png"), bbox_inches='tight')
plt.close()

# Plot components manually
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axes[0].plot(cissa_annual.t, cissa_annual.x_trend, color='blue', label='Extracted Trend')
axes[0].set_title("Univariate (Annual Mean): Long-Term Trend")
axes[0].legend()
axes[1].plot(cissa_annual.t, cissa_annual.x_periodic, color='orange', label='Extracted Periodicities')
axes[1].set_title("Univariate (Annual Mean): Periodic Components (Combined ENSO cycles)")
axes[1].legend()
axes[2].plot(cissa_annual.t, cissa_annual.x_noise, color='grey', alpha=0.7, label='Extracted Noise')
axes[2].set_title("Univariate (Annual Mean): Noise")
axes[2].legend()
plt.xlabel("Year")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "elnino_cissa_annual_components.png"), bbox_inches='tight')
plt.close()


print("\n--- 2. RUNNING UNIVARIATE CISSA (Continuous Monthly) ---")
# Flatten the entire dataset into one continuous line of months
x_continuous_monthly = X.flatten()
T_months = len(x_continuous_monthly)
t_months_continuous = np.arange(1, T_months + 1) / 12.0 + t_years[0] # Map back to fractional years for plotting

cissa_monthly = Cissa(t=t_months_continuous, x=x_continuous_monthly)
# L must be larger (e.g. 48 months = 4 years) to capture the 4 year cycle
cissa_monthly.fit(L=48)
cissa_monthly.post_run_frequency_time_analysis(data_per_period=12) # 12 data points per year
cissa_monthly.auto_cissa(L=48, plot_result=False, verbose=False)

cissa_monthly.plot_original_time_series()
plt.title("Univariate El Niño (Continuous Monthly)")
plt.xlabel("Year")
plt.ylabel("Temperature Anomaly")
plt.savefig(os.path.join(output_dir, "elnino_cissa_monthly_time_series.png"), bbox_inches='tight')
plt.close()

# Plot components manually
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axes[0].plot(cissa_monthly.t, cissa_monthly.x_trend, color='blue', label='Extracted Trend')
axes[0].set_title("Univariate (Continuous Monthly): Trend")
axes[0].legend()
axes[1].plot(cissa_monthly.t, cissa_monthly.x_periodic, color='orange', label='Extracted Periodicities')
axes[1].set_title("Univariate (Continuous Monthly): Periodic (Dominated by 1yr Seasonality)")
axes[1].legend()
axes[2].plot(cissa_monthly.t, cissa_monthly.x_noise, color='grey', alpha=0.7, label='Extracted Noise')
axes[2].set_title("Univariate (Continuous Monthly): Noise")
axes[2].legend()
plt.xlabel("Year")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "elnino_cissa_monthly_components.png"), bbox_inches='tight')
plt.close()


print("\n--- 3. RUNNING MULTIVARIATE M-CISSA (12 Months as Channels) ---")
mcissa = MCissa(t=t_years, x=X)
mcissa.fit(L=16)
mcissa.post_run_frequency_time_analysis(data_per_period=1)
mcissa.auto_cissa(L=16, plot_result=False, verbose=False)

mcissa.plot_original_time_series()
plt.title("Multivariate El Niño (12 Distinct Monthly Channels)")
plt.xlabel("Year")
plt.ylabel("Temperature Anomaly")
plt.savefig(os.path.join(output_dir, "elnino_mcissa_time_series.png"), bbox_inches='tight')
plt.close()

month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
comp_names = [
    "Comp 1: Decadal Trend",
    "Comp 2: Primary Dominant Cycle (e.g. Quasi-Quadrennial ENSO)",
    "Comp 3: Secondary Dominant Cycle"
]

mcissa.plot_components(num_components=3, variable_names=month_names, component_names=comp_names)
plt.savefig(os.path.join(output_dir, "elnino_mcissa_components.png"), bbox_inches='tight')
plt.close()

print("\nPlots generated successfully!")
