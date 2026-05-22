import numpy as np
import pytest
from pycissa.processing.cissa.cissa import Cissa
from tests.performance_accuracy_tests.accuracy_utils import evaluate_accuracy

def test_cissa_accuracy_synthetic():
    np.random.seed(42)
    T = 300
    t = np.arange(1, T + 1)

    # Ground truth components
    true_trend = 0.05 * t + 0.001 * t**2
    true_periodic = 2.5 * np.sin(2 * np.pi * t / 24) + 1.5 * np.cos(2 * np.pi * t / 12)
    true_noise = np.random.normal(0, 0.5, T)

    x = true_trend + true_periodic + true_noise

    # Run auto_cissa
    cissa = Cissa(t, x)
    cissa.auto_cissa(L=48, plot_result=False, verbose=False)

    # Evaluate Trend Extraction
    trend_metrics = evaluate_accuracy(true_trend, cissa.x_trend)
    assert trend_metrics['correlation'] > 0.95, f"Trend correlation too low: {trend_metrics['correlation']}"
    assert trend_metrics['mse'] < 2.0, f"Trend MSE too high: {trend_metrics['mse']}"

    # Evaluate Periodic Extraction
    periodic_metrics = evaluate_accuracy(true_periodic, cissa.x_periodic)
    assert periodic_metrics['correlation'] > 0.85, f"Periodic correlation too low: {periodic_metrics['correlation']}"
    assert periodic_metrics['mse'] < 1.0, f"Periodic MSE too high: {periodic_metrics['mse']}"

    # Evaluate Overall signal
    clean_signal = true_trend + true_periodic
    extracted_signal = cissa.x_trend + cissa.x_periodic
    overall_metrics = evaluate_accuracy(clean_signal, extracted_signal)
    assert overall_metrics['correlation'] > 0.95, f"Overall signal correlation too low: {overall_metrics['correlation']}"
