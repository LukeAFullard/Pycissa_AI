import numpy as np
from pycissa.processing.mcissa.mcissa import MCissa

t = np.arange(100)
x = np.random.rand(100, 2)
mcissa = MCissa(t, x)
mcissa.fit(10)
print(mcissa.psd.shape)
