import numpy as np
from math import pi
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy.special  import gamma
from sklearn.metrics import r2_score

def gamma_function_ps(x,tau0,exponent,amplitude):
    return amplitude*np.power(1 + np.power(2*pi*np.array(x)*tau0/exponent,2),-exponent)
def gamma_function_(t,tau0,exponent):
    return np.power(t,exponent-1)*np.exp(-exponent*t/tau0)/(np.power(tau0/exponent,exponent)*gamma(exponent))
def gamme_function_robust(x,my_freq,my_psd):
    return x[2]*np.power(1 + np.power(2*pi*np.array(my_freq)*x[0]/x[1],2),-x[1]) - my_psd
###############################################################################
###############################################################################
###############################################################################
def fit_gamma(my_freq,my_psd,array_length):
    try:popt, pcov = curve_fit(gamma_function_ps, my_freq, [x/1 for x in my_psd],method = 'trf', x_scale = [1, 1, 1])
    except RuntimeError as e:
        raise RuntimeError(f"{e}")
    condition_number = np.linalg.cond(pcov)
    psd_pred = gamma_function_ps(my_freq, *popt)
    r2 = r2_score(my_psd, psd_pred)
    tau0,exponent,amplitude = popt[0],popt[1],popt[2]

    t_ = np.linspace(1,array_length,array_length)
    h = gamma_function_(t_,tau0,exponent)

    #robust fitting
    x0 = np.ones(3)
    res_robust = least_squares(gamme_function_robust, x0, loss='huber', f_scale=0.01, args=(my_freq, my_psd))
    r2_robust = r2_score(my_psd, gamma_function_ps(my_freq, res_robust.x[0], res_robust.x[1],res_robust.x[2]))
    tau0_robust = res_robust.x[0]
    exponent_robust = res_robust.x[1]
    amplitude_robust = res_robust.x[2]

    print(r2_robust,tau0_robust,exponent_robust,amplitude_robust)
    return tau0,exponent, amplitude, r2
    #robust fitting
    # x0 = np.ones(2)
    # res_robust = least_squares(gamme_function_robust, x0, loss='huber', f_scale=0.01, args=(my_freq, my_psd))
    # r2_robust = r2_score(my_psd, gamma_function(my_freq, res_robust.x[0], res_robust.x[1]))
    # tau0_robust = res_robust.x[0]
    # exponent_robust = res_robust.x[1]

    # return tau0,exponent, r2,tau0_robust,exponent_robust, r2_robust