import numpy as np
import warnings
from pycissa.utilities.extendseries import extend_series

# Create a sequence with long constant gap
x = np.array([0., 1., 2., 0., 0., 0., 0., 0., 0., 0., 0., 3., 4., 5.]).reshape(-1, 1)

try:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        xe = extend_series(x, 'AR_LR', 5, 5)
        for warn in w:
            print(warn.message)
except Exception as e:
    import traceback
    traceback.print_exc()
