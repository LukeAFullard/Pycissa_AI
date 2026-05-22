import numpy as np
import matplotlib.pyplot as plt
from pycissa import Cissa

# Create a synthetic dataset with known ground truth
np.random.seed(42)
T = 300
t = np.arange(1, T + 1)

true_trend = 0.05 * t + 0.001 * t**2
true_periodic = 2.5 * np.sin(2 * np.pi * t / 24) + 1.5 * np.cos(2 * np.pi * t / 12)
true_noise = np.random.normal(0, 0.5, T)

x = true_trend + true_periodic + true_noise

print("Running univariate CISSA on synthetic data...")
cissa = Cissa(t, x)
cissa.auto_cissa(L=48, plot_result=True)

# Evaluate metrics
def evaluate(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    return mse, corr

trend_mse, trend_corr = evaluate(true_trend, cissa.x_trend)
periodic_mse, periodic_corr = evaluate(true_periodic, cissa.x_periodic)

print(f"\n--- Accuracy Metrics ---")
print(f"Trend MSE: {trend_mse:.4f}, Correlation: {trend_corr:.4f}")
print(f"Periodic MSE: {periodic_mse:.4f}, Correlation: {periodic_corr:.4f}")

plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.plot(t, x, label="Mixed Input", alpha=0.5)
plt.plot(t, true_trend + true_periodic, label="True Signal", color='black')
plt.title("CISSA Input Data")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, true_trend, label="True Trend", linestyle='dashed')
plt.plot(t, cissa.x_trend, label="Extracted Trend")
plt.title(f"Trend Comparison (Corr: {trend_corr:.3f})")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, true_periodic, label="True Periodic", linestyle='dashed')
plt.plot(t, cissa.x_periodic, label="Extracted Periodic")
plt.title(f"Periodic Comparison (Corr: {periodic_corr:.3f})")
plt.legend()

plt.tight_layout()
plt.savefig("examples/cissa/cissa_accuracy.png")
print("Saved plot to examples/cissa/cissa_accuracy.png")
