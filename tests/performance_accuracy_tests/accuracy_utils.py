import numpy as np

def calculate_mse(y_true, y_pred):
    """Calculates Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)

def calculate_correlation(y_true, y_pred):
    """Calculates Pearson Correlation Coefficient."""
    if np.var(y_true) == 0 or np.var(y_pred) == 0:
        return 0.0
    return np.corrcoef(y_true, y_pred)[0, 1]

def calculate_snr(signal, noise):
    """Calculates Signal-to-Noise Ratio (SNR) in dB."""
    power_signal = np.mean(signal ** 2)
    power_noise = np.mean(noise ** 2)
    if power_noise == 0:
        return np.inf
    return 10 * np.log10(power_signal / power_noise)

def evaluate_accuracy(y_true, y_pred):
    """Returns a dictionary of evaluation metrics."""
    return {
        'mse': calculate_mse(y_true, y_pred),
        'correlation': calculate_correlation(y_true, y_pred)
    }
