import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pycissa
from pycissa.processing.mcissa.mcissa import MCissa
t = np.arange(100)
x = np.random.randn(100, 2)
mcissa = MCissa(t, x)
mcissa.auto_cissa(L=10, grouping_type='monte_carlo')
print("Did it run cleanly?", hasattr(mcissa, "x_trend"))
