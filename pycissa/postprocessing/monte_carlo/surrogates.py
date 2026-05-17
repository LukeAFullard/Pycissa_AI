import numpy as np
import copy
###############################################################################
###############################################################################

def generate_random_permutation(x: np.ndarray)->np.ndarray:
    '''
    Function to randomly shuffle an input series

    Parameters
    ----------
    x : np.ndarray
        DESCRIPTION: Input series

    Returns
    -------
    x_copy : np.ndarray
        DESCRIPTION: Shuffled output series

    '''
    x_copy = copy.deepcopy(x)
    #get up random number generator
    rng = np.random.default_rng()
    rng.shuffle(x_copy)
    return x_copy

###############################################################################
###############################################################################
def generate_small_shuffle(x: np.ndarray,
                           A: float = 1.)->np.ndarray:
    '''
    Function to apply the Small Shuffle algorithm to a time series
    Nakamura, T., & Small, M. (2005). Small-shuffle surrogate data: Testing for dynamics in fluctuating data with trends. Physical Review E, 72(5), 056216.

    Parameters
    ----------
    x : np.ndarray
        DESCRIPTION: Input series
    A : float
        DESCRIPTION: Small shuffle amplitude. The default is 1.

    Returns
    -------
    x_copy : np.ndarray
        DESCRIPTION: Shuffled output series

    '''
    x_copy = copy.deepcopy(x)
    #get up random number generator
    rng = np.random.default_rng()
    gaussian_random_numbers = rng.standard_normal(len(x_copy))

    original_index = [x for x in range(0,len(x_copy))]
    perturbed_index = list(original_index + A*gaussian_random_numbers)
    sorted_perturbed = sorted(perturbed_index)
    new_index = [perturbed_index.index(x) for x in sorted_perturbed]

    #shuffle data according to the small shuffle index
    x_copy = x_copy[new_index]
    return x_copy


###############################################################################
###############################################################################
def generate_ar1_evenly(x:    np.ndarray,
                        seed: int = None):
    from statsmodels.tsa.arima_process import arma_generate_sample
    from statsmodels.tsa.arima.model import ARIMA
    if seed is not None:
            np.random.seed(seed)

    x_copy = copy.deepcopy(x)
    series_length = len(x_copy)
    signal_scale = np.std(x_copy)


    ###########################################################################
    #fit AR1 parameters
    ar1_mod = ARIMA(x_copy, order = (1,0,0), missing ='drop',trend ='ct').fit()
    rho = ar1_mod.arparams[0]
    if rho > 1.:
        print('Warning: rho > 1; setting to 1-eps^0.25')
        eps = np.spacing(1.0)
        rho = 1.0 - eps**(0.25)
    ar1_scale = signal_scale*np.sqrt(1-rho**2)
    ###########################################################################

    ###########################################################################
    #generate AR1 sample
    x_synthetic = arma_generate_sample(np.array([1,-rho]), np.array([1,0]), nsample=series_length, scale=ar1_scale, burnin=100)

    # standardize the generated AR1 sample and rescale to match original signal scale and mean
    x_synthetic = (x_synthetic - np.mean(x_synthetic)) / np.std(x_synthetic)
    x_synthetic = x_synthetic * signal_scale + np.mean(x_copy)

    ###########################################################################

    return x_synthetic


###############################################################################
###############################################################################


###############################################################################
###############################################################################

###############################################################################
###############################################################################