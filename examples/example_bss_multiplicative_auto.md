# M-CiSSA Blind Source Separation: Detecting Multiplicative Noise

This example demonstrates how to automatically detect if an artifact is multiplicative using the `test_if_multiplicative` utility.

M-CiSSA is inherently a linear algorithm designed to separate additive mixtures. Standard Singular Spectrum Analysis relies on finding finite linear recurrences.

When an artifact is **multiplicative**, it modulates the *amplitude (variance)* of the main signal rather than just shifting its mean. This causes cross-modulation (sidebands) in the frequency domain. If you feed a multiplicative mixture into a purely linear BSS pipeline without proper envelope extraction or demodulation techniques, the algorithm will struggle to cleanly strip the artifact because it does not exist as a simple additive wave.

## Automated Detection (Variance Correlation Test)

To detect if an artifact is multiplicative, we can check if the reference channel correlates with the **variance (or envelope)** of the mixed signal, rather than just the raw values.

The `test_if_multiplicative` utility function calculates a rolling standard deviation of the mixed signal and checks its Pearson correlation against the reference channel.

```python
from pycissa.preprocessing import test_if_multiplicative

is_mult, corr_raw, corr_std = test_if_multiplicative(mixed_signal, reference)

if is_mult:
    print("Warning: The artifact appears to modulate the amplitude of the signal.")
    print("Linear BSS may leave residual amplitude wobbles. Consider envelope demodulation.")
```

## Results

In the example script, the true signal's amplitude is modulated by a slow artifact.
- The **Variance Correlation Test** correctly flags it as multiplicative because the reference channel perfectly correlates with the rolling standard deviation of the mixed signal (`corr_std = 0.90`), but has near-zero correlation with the raw signal mean (`corr_raw = 0.00`).
- The script then runs a standard Linear M-CiSSA BSS on the data. As expected, because the artifact causes amplitude modulation (sidebands) rather than a simple additive frequency peak, linear BSS struggles to decouple it, leaving a persistent amplitude wobble in the recovered signal.

![Multiplicative Test Example](bss_multiplicative_auto_test.png)
