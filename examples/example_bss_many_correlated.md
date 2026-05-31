# Complex Additive (Many Correlated) BSS Example

This example demonstrates M-CiSSA's ability to clean a main signal that is heavily contaminated by a complex mixture of artifacts, using multiple reference channels that are highly correlated with each other and phase-shifted.

## The Challenge

In real-world scenarios (like EEG or industrial sensor networks), you often don't have a single clean reference for each artifact. Instead, you might have multiple reference sensors that all pick up different combinations of the same underlying interference sources (e.g., eye movements, muscle artifacts, power line hum, and baseline drift).

Furthermore, these sources might reach different sensors at slightly different times, introducing phase shifts (time delays).

## How M-CiSSA Solves It

M-CiSSA's Blind Source Separation (BSS) naturally handles these complexities because:
1. **Shared Dynamics:** It isolates shared dynamics into common spatial subcomponents.
2. **Phase Shifts:** It seamlessly handles time-shifted reference signals because time delays are captured as phase shifts via complex-valued spatial eigenvectors in the frequency domain.
3. **Monte Carlo Testing:** It performs a rigorous surrogate test to ensure it only removes components from the main signal that are statistically significant in the reference channels.

## The Code

We simulate a "Brain Activity" true signal (slow trend + alpha wave) and three distinct artifact sources (Drift, Hum, and Muscle).
We mix these artifacts into 4 reference channels with different weights and a time delay (phase shift) in one of the channels.
The main channel receives the true signal plus a complex mixture of all the artifacts.

```python
import numpy as np
import matplotlib.pyplot as plt
from pycissa.processing.mcissa.mcissa import MCissa

T = 300
t = np.arange(T)

# True main signal
true_signal = 3.0 * np.sin(2 * np.pi * t / 60.0) + 1.5 * np.sin(2 * np.pi * t / 15.0)

# Many correlated artifacts
art_src_1 = 5.0 * np.sin(2 * np.pi * t / 45.0)  # Drift
art_src_2 = 2.0 * np.sin(2 * np.pi * t / 6.0)   # Hum
art_src_3 = 3.0 * np.sin(2 * np.pi * t / 25.0)  # Muscle artifact

# Mix into 4 references
ref_1 = 1.0 * art_src_1 + 0.5 * art_src_2 + np.random.randn(T) * 0.2
ref_2 = 0.8 * art_src_1 + 1.2 * art_src_3 + np.random.randn(T) * 0.2
ref_3 = 0.2 * art_src_2 + 0.9 * art_src_3 + np.random.randn(T) * 0.2
ref_4 = 1.5 * (5.0 * np.sin(2 * np.pi * (t-5) / 45.0)) + np.random.randn(T) * 0.2

# Contaminate main signal
main_contamination = 1.2 * art_src_1 + 0.8 * art_src_2 + 1.5 * art_src_3 + 0.5 * ref_4
raw_mixed = true_signal + main_contamination + np.random.randn(T) * 0.5

X = np.column_stack([raw_mixed, ref_1, ref_2, ref_3, ref_4])

# We use Auto BSS on the 5-channel matrix
mcissa = MCissa(t, X)

# We use the actual Monte Carlo test to identify significance
mcissa.auto_blind_source_separation(
    L=60,
    main_index=0,
    K_surrogates=10,
    variance_threshold=0.01,
    alpha=0.05,
    trend_always_significant=False
)

recovered_signal = mcissa.x_cleaned
```

## Results

Even with 4 highly correlated reference channels and delayed signals, M-CiSSA accurately identifies the spectral power residing in the reference channels and strips it from the main signal.

As shown in the output plot, the Mean Squared Error (MSE) drops dramatically from ~54 (raw mixed vs true) to ~6.4 (recovered vs true), successfully reconstructing the underlying true signal. The recovered signal tracks the true signal closely, having successfully filtered out the complex additive artifacts.
