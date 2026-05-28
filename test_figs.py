import numpy as np
import statsmodels.api as sm
from pycissa.processing.mcissa.mcissa import MCissa

df = sm.datasets.elnino.load_pandas().data
X = df.iloc[:, 1:].values
t_years = df['YEAR'].values

mcissa = MCissa(t=t_years, x=X)
mcissa.fit(L=16)
mcissa.auto_cissa(L=16, plot_result=True, verbose=False)

print(mcissa.figures['mcissa'].keys())
first_key = list(mcissa.figures['mcissa'].keys())[0]
print(type(mcissa.figures['mcissa'][first_key]))
