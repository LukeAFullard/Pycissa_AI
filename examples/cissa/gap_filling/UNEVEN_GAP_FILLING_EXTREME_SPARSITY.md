# Handling Extreme Data Sparsity with M-CiSSA

When filling gaps in unevenly sampled time series, we often encounter datasets that are extremely sparse or heavily censored. If there are massive gaps relative to the sampling rate, the standard interpolation and spectral decomposition techniques (`pchip` + dynamic component selection) might completely fail to construct a meaningful representation, resulting in a very poor $R^2$ fit against the few true data points we do have.

## The Fallback Mechanism

In `pycissa`, both the univariate `fill_uneven_timeseries` and multivariate `m_fill_uneven_timeseries` functions are equipped with an automatic cascade of fallbacks to handle extreme sparsity:

1.  **Initial Attempt:** The algorithm tries the default or user-provided interpolation method (usually `pchip`) and performs a standard spectral decomposition, selecting only the most significant components (dropping the smallest variance).
2.  **First Fallback (Linear):** If the fit is very poor (i.e., $R^2 <$ `r2_warning_threshold`), the algorithm automatically restarts the gap-filling process using simpler `linear` interpolation.
3.  **Second Fallback (Retain All Components):** If the data is so sparse that even linear interpolation yields a poor fit (because the significant component selection strips too much variance from an already starved signal), it triggers a final fallback. It retries using `linear` interpolation but explicitly disables component dropping (`component_selection_method='none'`). This preserves whatever faint signal exists in the sparse points.
4.  **Warning:** If all fallbacks fail to achieve the required $R^2$ threshold, the algorithm gracefully returns the best attempt and issues a single warning.

## Example Python Script

The accompanying script `run_extreme_sparsity_example.py` demonstrates this mechanism. We generate a synthetic, noisy sine wave and intentionally punch massive, structural gaps into it, leaving only a few isolated points (extreme sparsity).

## Results

As shown in the generated plot `extreme_sparsity_fallback.png`, despite the massive gaps and sparsity, the algorithm utilizes the secondary fallback (`linear` interpolation + `none` component selection) to piece together a reasonable estimation of the underlying dynamics without throwing fatal errors.
