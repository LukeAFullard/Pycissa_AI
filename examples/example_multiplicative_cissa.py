import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from pycissa.processing.cissa.cissa import Cissa
from pycissa.preprocessing import MultiplicativeTransformer

# 1. Create a signal with Multiplicative Seasonality
T = 400
t = np.arange(T)
# Base trend that grows linearly
base_trend = 10.0 + 0.05 * t
# Seasonality whose amplitude scales with the trend (multiplicative)
seasonality = 1.0 + 0.5 * np.sin(2 * np.pi * t / 20.0)
# True signal = Trend * Seasonality
raw_signal = base_trend * seasonality + np.random.randn(T) * 0.5

print("--- Standard CiSSA (Linear) ---")
# If we run linear CiSSA, it will struggle because the amplitude of the seasonal component changes over time.
# The spectral power of the seasonal component spreads across multiple frequencies (modulation).
cissa_linear = Cissa(t, raw_signal)
cissa_linear.fit(L=100)
cissa_linear.auto_detrend(trend_threshold=0.99)
trend_linear = cissa_linear.x_trend

print("\n--- Log-Transformed CiSSA (Multiplicative) ---")
# By taking the log, we linearize the problem: log(Trend * Seasonality) = log(Trend) + log(Seasonality)
transformer = MultiplicativeTransformer()
log_signal = transformer.fit_transform(raw_signal)

cissa_log = Cissa(t, log_signal)
cissa_log.fit(L=100)
cissa_log.auto_detrend(trend_threshold=0.99)

# We now have the components in log space. To recover them, we must invert the transform.
# The recovered main signal is exp(log(trend) + log(seasonal))
# If we just want the isolated trend in the original scale:
trend_log_space = cissa_log.x_trend
# We invert the trend specifically
trend_recovered = transformer.inverse_transform(trend_log_space, col_idx=0)

plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.title("Raw Signal (Multiplicative Seasonality)")
plt.plot(t, raw_signal, label="Raw Data (Amplitude grows with trend)", color='lightgray')
plt.plot(t, base_trend, label="True Base Trend", color='black', linestyle='--')
plt.legend()

plt.subplot(3, 1, 2)
plt.title("Standard Linear CiSSA Recovery")
plt.plot(t, raw_signal, label="Raw Data", color='lightgray')
plt.plot(t, base_trend, label="True Trend", color='black', linestyle='--')
plt.plot(t, trend_linear, label="Extracted Trend (Struggles with amplitude)", color='red')
plt.legend()

plt.subplot(3, 1, 3)
plt.title("Multiplicative (Log) CiSSA Recovery")
plt.plot(t, raw_signal, label="Raw Data", color='lightgray')
plt.plot(t, base_trend, label="True Trend", color='black', linestyle='--')
plt.plot(t, trend_recovered, label="Extracted Trend (Perfect tracking)", color='blue')
plt.legend()

plt.tight_layout()
plt.savefig("examples/cissa_multiplicative_example.png")
print("\nPlot saved as 'examples/cissa_multiplicative_example.png'")
