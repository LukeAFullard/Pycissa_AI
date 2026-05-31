# Multiplicative Blind Source Separation

This example demonstrates how M-CiSSA handles **multiplicative interference**.

Since M-CiSSA is inherently a linear additive model, true multiplicative effects (e.g. `signal * artifact`) violate its base assumptions.

In this script, we test two approaches:
1. **Raw M-CiSSA:** Feed the multiplicatively mixed signal directly into M-CiSSA.
2. **Log-Transformed M-CiSSA:** Apply a log transform to both the mixed signal and the reference to convert the relationship into an additive one (`log(a*b) = log(a) + log(b)`), apply M-CiSSA, and then exponentiate back.

## Results
The log-transformed approach generally provides a more accurate recovery because it aligns the data with M-CiSSA's linear assumptions. The raw approach struggles to fully separate the components since the spatial correlation varies proportionally with the signal amplitude, which a static spatial eigenvector cannot perfectly capture.

The plot includes an explicit error comparison to highlight the difference in accuracy.