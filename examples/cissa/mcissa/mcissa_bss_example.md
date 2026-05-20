# M-CiSSA Blind Source Separation Example

This example demonstrates an advanced use case of Multivariate Circulant Singular Spectrum Analysis (M-CiSSA): **Blind Source Separation (BSS) / Reference Channel Filtering**.

Unlike standard univariate SSA which processes a single channel, M-CiSSA jointly processes multiple variables. This allows us to use known reference signals to "pull out" or filter out corresponding variance from a mixed target channel, isolating the unknown or leftover signals.

## The Scenario

Imagine we have a sensor (Variable 0) that records a complex mixture of three sources:
1. A slow trend ($s_1$)
2. A low-frequency oscillation ($s_2$)
3. A high-frequency oscillation ($s_3$)

We want to isolate $s_3$. We don't know the exact mathematical form of $s_3$, but we *do* have reference sensors that perfectly capture $s_1$ and $s_2$ independently.

We can feed all three channels into M-CiSSA:
* **Variable 0**: $s_1 + s_2 + s_3$ (The Mixed Target)
* **Variable 1**: $s_1$ (Reference 1)
* **Variable 2**: $s_2$ (Reference 2)

## Running the Example

To run the verification script, execute the following from this directory:

```bash
python run_mcissa_bss_example.py
```

## Explanation of the Code

First, we generate our three pure signals and construct our multivariate dataset `X`:

```python
import numpy as np
import matplotlib.pyplot as plt
from pycissa.processing.mcissa.mcissa import MCissa

N = 300
t = np.arange(N)

# 1. Generate three independent pure signals
s1 = np.linspace(0, 10, N)  # Trend
s2 = 3.0 * np.sin(2 * np.pi * t / 20)  # Low freq oscillation
s3 = 1.5 * np.sin(2 * np.pi * t / 5)   # High freq oscillation

# 2. Mix them together (this is our target channel)
mixed = s1 + s2 + s3

# 3. Create a multivariate dataset containing the mixed signal (Var 0)
# and two of the pure signals as "reference" channels (Var 1 and Var 2)
X = np.column_stack((mixed, s1, s2))
```

We then initialize and fit M-CiSSA:

```python
L = 100
mcissa = MCissa(t=t, x=X)
mcissa.fit(L=L)
```

Because M-CiSSA jointly diagonalizes the dataset, the variance associated with $s_1$ and $s_2$ gets strongly pulled into joint components that correlate with the reference channels. The leftover variance in the mixed channel (Variable 0) naturally isolates $s_3$.

We can extract the components for Variable 0 and find the ones corresponding to our leftover signal:

```python
# The reconstructed components for variable 0 are in mcissa.Z_stacked[:, 0, :]
components_var0 = mcissa.Z_stacked[:, 0, :]

# In this synthetic example, we find the s3 components using correlation.
# In a real scenario, you would look at the eigenvalue groupings or frequencies.
correlations = np.array([np.abs(np.corrcoef(components_var0[:, i], s3)[0, 1]) for i in range(components_var0.shape[1])])

# An oscillation splits into 2 conjugate components, so we take the top 2
top_indices = np.argsort(correlations)[-2:]

# Reconstruct s3 from these two components (truncate to N length)
extracted_s3 = np.sum(components_var0[:N, top_indices], axis=1)
```

Finally, we calculate the error and plot the result:

```python
error = s3 - extracted_s3
max_err = np.max(np.abs(error))
print(f"Max extraction error for s3: {max_err}")
```

```
Max extraction error for s3: 0.10245922430854404
```

## Results

When you run the script, it produces `mcissa_bss_extraction.png`. The output demonstrates that M-CiSSA successfully utilized the cross-variable relationships to filter out the known signals and cleanly extract the hidden $s_3$ signal with only minor boundary errors (maximum error ~0.1 on a signal with amplitude 1.5).