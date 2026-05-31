import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from pycissa.processing.cissa.overlap_cissa import OverlapCissa
from pycissa.preprocessing import MultiplicativeTransformer

T = 2000
t = np.arange(T)

true_trend = 10.0 + 0.01 * t + 0.00002 * t**2
seasonality = 1.0 + 0.4 * np.sin(2 * np.pi * t / 50.0)
raw_signal = true_trend * seasonality + np.random.randn(T) * 0.5

print("--- Overlap CiSSA on Multiplicative Data ---")

transformer = MultiplicativeTransformer()
log_signal = transformer.fit_transform(raw_signal)

L = 100
L_bar = L // 2
q = 200
Z = q + 2 * L_bar

overlap_cissa = OverlapCissa(t, log_signal, Z=Z, q=q, L=L, L_bar=L_bar)
overlap_cissa.fit()

# I must be a dict
I = {'trend': [0]}
overlap_cissa.post_group_manual(I=I)

t_overlap = overlap_cissa.t
# The grouped components for OverlapCissa are stored in the results dictionary
trend_log_space = overlap_cissa.results['cissa']['manual']['rc']['trend'].flatten()

trend_recovered = transformer.inverse_transform(trend_log_space, col_idx=0)
true_trend_overlap = true_trend[:len(t_overlap)]

plt.figure(figsize=(12, 6))
plt.title("Overlap CiSSA: Multiplicative Long Timeseries Extraction")
plt.plot(t, raw_signal, label="Raw Long Timeseries", color='lightgray')
plt.plot(t_overlap, true_trend_overlap, label="True Underlying Trend (Sliced)", color='black', linestyle='--')
plt.plot(t_overlap, trend_recovered, label="Extracted Trend (Log Overlap CiSSA)", color='blue')
plt.legend()
plt.tight_layout()
plt.savefig("examples/cissa_multiplicative_overlap_example.png")
print("\nPlot saved as 'examples/cissa_multiplicative_overlap_example.png'")
