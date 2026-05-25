# Centering Uneven Monthly Data (Multivariate vs Univariate)

Often time series data like monthly observations aren't measured on exactly the same day every month. Because standard spectral methods require data on an evenly spaced grid, we must 'center' or align this jittery data to a strict regular frequency (e.g. exactly every 30 days).

`MCissa` simplifies this operation with `pre_fill_uneven_timeseries`. Because it supports **multivariate** processing, `MCissa` can leverage information from cross channels to confidently map uneven data points and fill large missing data periods.

Below are three examples demonstrating why M-CiSSA dramatically outperforms univariate methods when filling missing intervals during the centering process. In all scenarios, we generate 5 years of synthetic monthly data (60 months) with random jitter of ±5 days, and simulate a large missing data block (missing months 20, 21, 22, 40, and 41).

---

## Scenario 1: Highly Correlated Seasonality
In this baseline scenario, we have a simple annual cycle mapped across 3 highly correlated channels with differing amplitudes and noise. When entire blocks of data are missing from Channel 1, Univariate CiSSA struggles to interpolate across the gap because it only relies on its own historic periodicity. M-CiSSA recognizes the shared spatial component and actively utilizes Channel 2 and Channel 3 to project the missing amplitude perfectly.

**Channel 1 True Gap Recovery RMSE**
* **Univariate:** 3.39
* **Multivariate:** 1.20

![Scenario 1 Plot](monthly_centering_scenario_1.png)

---

## Scenario 2: Mixed Frequencies and Trends
Real-world data often involves complex combinations of differing periodicities and aggressive linear trends. Here, Channel 1 is purely an annual cycle, while Channel 2 combines annual, semi-annual, and a steep trend. Channel 3 inverts the semi-annual cycle.

Because M-CiSSA decomposes these components collectively, it separates the shared "annual" component away from the steep trend and the semi-annual cycles. It cleanly extracts exactly what it needs from the other channels to patch Channel 1.

**Channel 1 True Gap Recovery RMSE**
* **Univariate:** 3.25
* **Multivariate:** 1.35

![Scenario 2 Plot](monthly_centering_scenario_2.png)

---

## Scenario 3: Phase-Shifted Signals
In many physical systems, one channel might be a "delayed" reaction to another. In this scenario, we use the exact same frequency, but Channel 2 is phase-shifted (delayed) by 30 days, and Channel 3 is advanced by 60 days.

M-CiSSA natively handles time delays because they manifest as phase shifts inside complex-valued spatial eigenvectors in the frequency domain. Therefore, it mathematically understands that the channels are perfectly correlated (just offset in time), allowing it to accurately "look ahead" or "look behind" to fill the gaps.

**Channel 1 True Gap Recovery RMSE**
* **Univariate:** 4.24
* **Multivariate:** 1.87

![Scenario 3 Plot](monthly_centering_scenario_3.png)


---

## Scenario 4: Gap Size Sensitivity
How does the error scale as the gap size increases? In this scenario, we simulate a sensor failure where ONLY Channel 1 stops recording for consecutive months (ranging from 2 to 6 months missing), while Channels 2 and 3 continue to record.

Because Univariate only sees its own past history, its error blows up rapidly when predicting further into the future (or bridging a wide gap). However, because Multivariate M-CiSSA utilizes the actively recorded data in Channels 2 and 3, its error remains low and exceptionally stable even as the gap size expands!

![Scenario 4 Plot](monthly_centering_scenario_4.png)

---

## Scenario 5: Pure Centering Test (No Missing Months)
What if the data is just slightly unevenly sampled (jittered), but no months are actually missing?

In this scenario, we evaluate pure centering performance. The results are exactly identical between Univariate and Multivariate M-CiSSA.
* **Univariate RMSE:** 0.46
* **Multivariate RMSE:** 0.46

**Why?** Because when there are no gaps that cross the `gap_threshold`, the internal algorithms for both rely purely on standard mathematical interpolation (cubic spline) to snap the known measurements onto the evenly spaced grid points. Without any gaps to recover using spectral decomposition, cross-channel correlation is not invoked!
