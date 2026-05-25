import re

with open('pycissa/preprocessing/gap_fill/uneven_gap_filling.py', 'r') as f:
    content = f.read()

# We want to replace the whole Univariate Optimization loop
# I will just write a new version of the file completely since both univariate and multivariate need restructuring.
