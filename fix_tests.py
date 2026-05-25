import re

with open('pycissa/preprocessing/gap_fill/uneven_gap_filling.py', 'r') as f:
    content = f.read()

# Make it backwards compatible with `kwargs.get('component_selection_method')` for m_fill_uneven_timeseries
content = content.replace("model.fit(L=L, **{k: v for k, v in kwargs.items() if k not in ['outliers', 'gap_threshold', 'dt', 'center_data', 'multivariate', 'estimate_error', 'verbose']})", "model.fit(L=L, **{k: v for k, v in kwargs.items() if k not in ['outliers', 'gap_threshold', 'dt', 'center_data', 'multivariate', 'estimate_error', 'verbose', 'component_selection_method', 'eigenvalue_proportion', 'alpha']})")

with open('pycissa/preprocessing/gap_fill/uneven_gap_filling.py', 'w') as f:
    f.write(content)
