import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from pycissa.processing.mcissa.mcissa import MCissa
from pycissa.preprocessing import MultiplicativeTransformer

T = 400
t = np.arange(T)

# 1. Create a multivariate signal with shared multiplicative effects
# Shared underlying trend across both channels
shared_trend = 10.0 + 0.05 * t
# Channel 1: Strong seasonality
chan1_seasonality = 1.0 + 0.5 * np.sin(2 * np.pi * t / 20.0)
# Channel 2: Different seasonality, phase shifted, lower amplitude
chan2_seasonality = 1.0 + 0.3 * np.sin(2 * np.pi * t / 35.0 + np.pi/4)

raw_chan1 = shared_trend * chan1_seasonality + np.random.randn(T) * 0.2
raw_chan2 = shared_trend * chan2_seasonality + np.random.randn(T) * 0.2
X = np.column_stack([raw_chan1, raw_chan2])

print("--- Log-Transformed M-CiSSA (Multiplicative) ---")
# Safely log-transform all channels in the 2D array
transformer = MultiplicativeTransformer()
X_log = transformer.fit_transform(X)

# Run M-CiSSA on the linearized data
mcissa = MCissa(t, X_log)
mcissa.fit(L=100)
# Extract the shared trend using multivariate auto_detrend
mcissa.auto_detrend(trend_threshold=0.99)

# We have extracted the trend in log space. Now we invert it for each channel.
trend_chan1_recovered = transformer.inverse_transform(mcissa.x_trend[:, 0], col_idx=0)
trend_chan2_recovered = transformer.inverse_transform(mcissa.x_trend[:, 1], col_idx=1)

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.title("Channel 1: Multiplicative Trend & Recovery")
plt.plot(t, X[:, 0], label="Raw Channel 1 (Multiplicative)", color='lightgray')
plt.plot(t, shared_trend, label="True Shared Trend", color='black', linestyle='--')
plt.plot(t, trend_chan1_recovered, label="Extracted Trend (Log M-CiSSA)", color='blue')
plt.legend()

plt.subplot(2, 1, 2)
plt.title("Channel 2: Multiplicative Trend & Recovery")
plt.plot(t, X[:, 1], label="Raw Channel 2 (Multiplicative)", color='lightgray')
plt.plot(t, shared_trend, label="True Shared Trend", color='black', linestyle='--')
plt.plot(t, trend_chan2_recovered, label="Extracted Trend (Log M-CiSSA)", color='red')
plt.legend()

plt.tight_layout()
plt.savefig("examples/mcissa_multiplicative_example.png")
print("\nPlot saved as 'examples/mcissa_multiplicative_example.png'")
