import numpy as np

###############################################################################
###############################################################################
def generate_grouping(psd,L,trend = False):
    '''
    Function to group signal into its frequency and cell reference.

    Parameters
    ----------
    psd : numpy array
        DESCRIPTION: Column vector with the estimated power spectral density at frequencies w(k)=(k)/L, k=0,2,...,L-1, obtained with CiSSA.
    L : int
        DESCRIPTION: Window length.

    Returns
    -------
    mydict : dict
        DESCRIPTION: Dictionary of frequency (keys) and cell reference (values).
                     Frequency has units of 1/(number of timesteps).
                     For example, if a frequency of 0.083333333 is returned, this is equivalent to 1/0.083333333 = 12 time steps.

    '''
    number_of_groups = int(len(psd)/2 - 1)
    mydict = {} #In this dict, the keys will be the frequency, the value is the cell number
    for group_i in range(1,number_of_groups+1):  #NOTE: We start from 1 as we don't want k = 0, which has period = infinity. Effectively then, this detrends the series
        mydict.update({group_i/L:np.array([int(group_i)])})
    if trend:
        mydict.update({'trend':np.array([0])})
    return mydict
###############################################################################
###############################################################################
def group(Z,psd,I,season_length = 1, cycle_length = [1.5,8], include_noise = True):
    '''
    GROUP - Grouping step of CiSSA.  https://doi.org/10.1016/j.sigpro.2020.107824.

    This function groups the reconstructed components by frequency
    obtained with CiSSA into disjoint subsets and computes the share of the
    corresponding PSD.

    Syntax:     [rc, sh, kg] = group(Z,psd,I)

    Conversion from Matlab, https://github.com/jbogalo/CiSSA


    Parameters
    ----------
    Z : numpy array/matrix
        DESCRIPTION: Matrix whose columns are the reconstructed components by frequency obtained with CiSSA.
    psd : numpy column vector
        DESCRIPTION: Column vector with the estimated power spectral density at frequencies w(k)=(k)/L, k=0,2,...,L-1, obtained with CiSSA.
    I : multiple
        DESCRIPTION:
             Four options:
             1) A positive integer. It is the number of data per year in
             time series. The function automatically computes the
             trend (oscillations with period greater than 8 years), the
             business cycle (oscillations with period between 8 & 1.5 years)
             and seasonality.
             2) A dictionary. Each value contains a numpy row vector with the desired
             values of k to be included in a group, k=0,1,2,...,L/2-1. The function
             computes the reconstructed components for these groups.
             3) A number between 0 & 1. This number represents the accumulated
             share of the psd achieved with the sum of the share associated to
             the largest eigenvalues. The function computes the original
             reconstructed time series as the sum of these components.
             4) A number between -1 & 0. It is a percentile (in positive) of
             the psd. The function computes the original reconstructed time
             series as the sum of the reconstructed componentes by frequency
             whose psd is greater that this percentile.
    season_length : int, optional
        DESCRIPTION: The default is 1. Only used for case 1. when I = A positive integer. Can be modified in the case that the season is not equal to the number of data per year. For example, if a "season" is 2 years, we enter I = 365 (for days in year) and 2 for season_length because the season is 365*2, or data_per_year*season_length.
    cycle_length : list, optional
        DESCRIPTION: The default is [1.5,8]. List of longer term cycle periods. Only used for case 1.
    include_noise : bool, optional
        DESCRIPTION: The default is True. Output noise as a vector component or not. Only used for case 1.


    Returns
    -------
    rc : numpy array
        DESCRIPTION: Matrix whose columns are the reconstructed components for each
             group or subset of frequencies. In the case of economic time series
             the trend, business cycle and seasonality are in the first, second
             and third columns, respectively.
    sh : numpy array
        DESCRIPTION: Column vector with the share() of the psd for each group.
    kg : dict
        DESCRIPTION: Dictionary where each entry contains a row vector with the values
             of k belonging to a group. Option 1) produces 3 groups, option 2)
             gives the goups introduced by the user and options 3) and 4) produce
             a single group. In option 3), the values of k are sorted according
             to the share in total psd of their corresponding eigenvalues.

    -------------------------------------------------------------------------
    References:
    [1] Bógalo, J., Poncela, P., & Senra, E. (2021).
        "Circulant singular spectrum analysis: a new automated procedure for signal extraction".
         Signal Processing, 179, 107824.
        https://doi.org/10.1016/j.sigpro.2020.107824.
    -------------------------------------------------------------------------

    '''

    import numpy as np

    ###########################################################################
    # 0) psd size checking
    ###########################################################################
    rows, cols = psd.shape
    if cols != 1:
        raise ValueError(f'Input "psd" should be a column vector (i.e. only contain a single column). The size of x is ({rows},{cols})')
    ###########################################################################


    ###########################################################################
    # 1) Checking the input arguments
    ###########################################################################
    # Length and number of the reconstruted series
    T, F = Z.shape

    # Window length
    L = len(psd)


    # Type and value of input argument #3
    if type(I) is dict:
        opc = 2;
    elif type(I) == int or type(I) == float:
        if ((I-np.floor(I))==0) & (I>0):
            opc = 1
        elif (0<I)&(I<1):
            opc = 3
        elif (-1<I)&(I<0):
            opc = 4
        else:
            raise ValueError('***  Input argument #3 (I): Value ({I}) not valid  ***')

    else:
        raise ValueError(f'***  Input argument #3 ({I}): Type ({type(I)}) not valid  ***')


    #NOTE, Matlab version uses switch statements, but this was only implemented in Python in 3.10, so here we use if else statements for compatibility with older Python versions. Definately could improve this in the future...
    if opc == 1:
        # Proportionality of L
        if np.mod(L,I):
            raise ValueError(f'***  L is not proportional to the number of data per year  (modulo of L/I = {np.mod(L,I)}) ***');

    elif opc == 2:
        # Number of groups
        G = len(I.keys())
        if G>F:
            raise ValueError(f'***  The number of groups ({G}) is greater than the number of frequencies ({F})  ***')

        key_names = list(I.keys())
        # Disjoint groups
        for j in range(1,G):
            for m in range(j+1,G+1):
                if len(np.intersect1d(I[key_names[j-1]],I[key_names[m-1]]))>0:
                    raise ValueError(f'''***  The groups are not disjoint  ***
                                     {key_names[j-1]}: {I[key_names[j-1]]}
                                     {key_names[m-1]}: {I[key_names[m-1]]}                        ''');


    ###########################################################################


    ###########################################################################
    # 2) PSD for frequencies <= 1/2
    ###########################################################################
    if np.mod(L,2):
        pzz = np.append(psd[0], 2*psd[1:F])
    else:
        pzz = np.append(
            np.append(psd[0], 2*psd[1:F-1]),
            psd[F-1]
            )

    ###########################################################################


    ###########################################################################
    # 3) Indexes k for each group
    ###########################################################################
    #NOTE, Matlab version uses switch statements, but this was only implemented in Python in 3.10, so here we use if else statements for compatibility with older Python versions. Definately could improve this in the future...
    if opc == 1:
        # Number of groups
        G = 3
        if include_noise:
            G=4

        # Number of data per year
        s = I
        # Inizialitation of empty dict
        kg = {}
        # Seasonality    season_length cycle_length
        kg.update({'seasonality': L*np.arange(1,np.floor(s/2)+1)/(season_length*s)})

        # Long term cycle
        kg.update({'long term cycle': np.arange(max(1,np.floor(L/(cycle_length[1]*s)+1))-1,min(F-1,np.floor(L/(cycle_length[0]*s)+1)),dtype=int)})
        # Trend
        kg.update({'trend': np.arange(0,kg['long term cycle'][0])})

        #Noise: The left over frequencies
        if include_noise:
            current_k = []
            for index_j in kg.values():
                current_k = current_k + [int(x) for x in index_j]
            missing_k = [x for x in range(0,int(np.floor(L/2))) if x not in current_k]
            kg.update({'noise': np.array(missing_k)})


    elif opc == 2:
        # Groups
        kg = I.copy()
    elif opc == 3:

        # Number of groups
        G = 1
        # Inizialitation of list array
        kg = {}
        # Eigenvalues in decreasing order
        psor=np.sort(pzz)[::-1]
        ks=np.argsort(-pzz)
        # Cumulative share in percentage
        pcum = 100*np.cumsum(psor)/sum(psd)
        # Group for the reconstructed time series
        kg.update({1: ks[np.arange(0,len(ks[pcum<100*I])+1)]})
    elif opc == 4:
        # Number of groups
        G = 1
        # Inizialitation of cell array
        kg = {}
        # All k values
        ks = np.arange(0,F)
        # Group for the reconstructed time series

        kg.update({1:  ks[pzz>np.percentile(pzz,-100*I)]    })
    else:
        raise ValueError(f'*** Value of opc ({opc}) is incorrect. Something has gone terribly wrong... ***')
    ###########################################################################


    ###########################################################################
    # 4) Output arguments
    ###########################################################################
    # Inizialitation of output arguments
    # rc = np.zeros((T,G))
    # sh = np.zeros((G,1))


    # # Computing output arguments
    rc = {}
    sh = {}
    psd_sh = {}
    for index_j,key_j in enumerate(kg.keys()):
        # % Reconstructed component for each group
        indx=[int(x) for x in kg[key_j]]
        rc.update({key_j:np.sum(Z[:,indx],axis = 1, keepdims = True)})

        # % Psd share for each group
        sh.update({key_j: 100*np.sum(pzz[indx])/np.sum(pzz)})
        psd_sh.update({key_j: np.sum(pzz[indx])})


    return rc, sh, kg, psd_sh
###############################################################################
###############################################################################

###############################################################################
###############################################################################

###############################################################################
###############################################################################

###############################################################################
###############################################################################
#NOTE:THE FUNCTIONS BELOW ARE USED TO DROP SOME SSA COMPONENTS.
#     FOR EXAMPLE, FOR GAP FILLING, REMOVING NON-AUTOREGRESSIVE COMPONENTS, ETC

def find_smallest_values(d: dict,
                         n: int) -> list:
    '''
    Function find the n smallest values in a dictionary and return the corresponding dictionary keys

    Parameters
    ----------
    d : dict
        DESCRIPTION: Input dictionary
    n : int
        DESCRIPTION: Number of keys to return

    Returns
    -------
    list
        DESCRIPTION: List of keys of the n smallest values in the input dictionary

    '''
    # sort the dictionary items by value in ascending order
    sorted_items = sorted (d.items (), key=lambda x: x [1])
    # get the first n items from the sorted list
    smallest_items = sorted_items [:n]
    # get the keys from the smallest items
    smallest_keys = [k for k, v in smallest_items]
    # return the list of keys
    return smallest_keys
###############################################################################
###############################################################################
def drop_smallest_n_components(Z:                        np.ndarray,
                               psd:                      np.ndarray,
                               L:                        int,
                               number_of_groups_to_drop: int,
                               include_trend:            bool = True,
                               ) -> np.ndarray:
    '''
    Function to organise the extracted cissa components by their share of the power spectral density.
    The function will then drop a pre-defined number of these components in an effort to remove "noise".
    The smallest components are the ones which are dropped.

    Parameters
    ----------
    Z : np.ndarray
        DESCRIPTION: Output CiSSA results (from the run_cissa function).
    psd : np.ndarray
        DESCRIPTION: estimation of the the circulant matrix power spectral density
    L : int
        DESCRIPTION: CiSSA window length.
    number_of_groups_to_drop : int
        DESCRIPTION: Number of extracted components to drop.
    include_trend : bool, optional
        DESCRIPTION. The default is True. If False, the trend will always be removed.

    Returns
    -------
    new_array : np.ndarray
        DESCRIPTION: Reconstructed time-series with number_of_groups_to_drop components removed.
    '''
    myfrequencies = generate_grouping(psd,L, trend=include_trend)
    rc, sh, kg, psd_sh = group(Z,psd,myfrequencies)
    smallest_keys = find_smallest_values(sh, number_of_groups_to_drop)
    rckeys = [x for x in rc.keys() if x not in smallest_keys]
    new_array = np.zeros(rc[next(iter(rc))].shape)
    for key_i in rckeys: #iterate through the components to rebuild the signal
        new_array += rc[key_i]

    return new_array


def classify_smallest_n_components(Z:                    np.ndarray,
                               psd:                      np.ndarray,
                               L:                        int,
                               number_of_groups_to_drop: int,
                               include_trend:            bool = True,
                               ) -> np.ndarray:
    '''
    Function to organise the extracted cissa components by their share of the power spectral density.
    The function will then classify a pre-defined number of these components as "noise".
    The trend and remaining "periodic" signal components will be separately be classified.

    Parameters
    ----------
    Z : np.ndarray
        DESCRIPTION: Output CiSSA results (from the run_cissa function).
    psd : np.ndarray
        DESCRIPTION: estimation of the the circulant matrix power spectral density.
    L : int
        DESCRIPTION: CiSSA window length.
    number_of_groups_to_drop : int
        DESCRIPTION: Number of extracted components to drop.
    include_trend : bool, optional
        DESCRIPTION. The default is True. If False, the trend will always be removed.

    Returns
    -------
    trend : list
        DESCRIPTION: List of array locations for the trend component
    periodic : list
        DESCRIPTION: List of array locations for the periodic components
    noise : list
        DESCRIPTION: List of array locations for the noise components
    '''
    myfrequencies = generate_grouping(psd,L, trend=include_trend)
    rc, sh, kg, psd_sh = group(Z,psd,myfrequencies)
    smallest_keys = find_smallest_values(sh, number_of_groups_to_drop)
    trend    = [kg['trend'][0]]
    periodic = sorted([kg[x][0] for x in rc.keys() if x not in smallest_keys and x != 'trend'])
    noise    = sorted([kg[x][0] for x in rc.keys() if x in smallest_keys])

    return trend,  periodic, noise
###############################################################################
###############################################################################
def drop_smallest_proportion_psd(Z,psd,eigenvalue_proportion):
    '''
    Drop components based on either the cumulative psd being larger than a supplied threshold, or on indvidual components supplying a threshold to the psd.
    For example, we may choose the eigenvalue_proportion = 0.9 which will add components to the reconstructed signal (from largest to smallest) until 90% of the psd is achieved. The remaining components (which make up the remaining 10%) will be dropped.
    Alternatively, we may choose eigenvalue_proportion = -0.1 which means only components that comtribute at least 10% to the psd will be kept.


    Parameters
    ----------
    Z : np.ndarray
        DESCRIPTION: Output CiSSA results (from the run_cissa function).
    psd : np.ndarray
        DESCRIPTION: estimation of the the circulant matrix power spectral density
    eigenvalue_proportion : TYPE
        DESCRIPTION: Option for how to select components to keep. There are three options, although the main purpose of this function is options 2 and 3:
                1) A dictionary. Each value contains a numpy row vector with the desired
                values of k to be included in a group, k=0,1,2,...,L/2-1. The function
                computes the reconstructed components for these groups.
                2) A number between 0 & 1. This number represents the accumulated
                share of the psd achieved with the sum of the share associated to
                the largest eigenvalues. The function computes the original
                reconstructed time series as the sum of these components.
                3) A number between -1 & 0. It is a percentile (in positive) of
                the psd. The function computes the original reconstructed time
                series as the sum of the reconstructed componentes by frequency
                whose psd is greater that this percentile.

    Returns
    -------
    new_array : np.ndarray
        DESCRIPTION: Reconstructed time-series with number_of_groups_to_drop components removed.

    '''
    rc, sh, _, _ = group(Z,psd,eigenvalue_proportion)

    new_array = np.zeros(Z[:,0:1].shape)

    for key_i in rc.keys(): #iterate through the components to rebuild the signal
        new_array += rc[key_i]
    return new_array


def classify_smallest_proportion_psd(Z,psd,L,eigenvalue_proportion):
    '''
    Classify components based on either the cumulative psd being larger than a supplied threshold, or on indvidual components supplying a threshold to the psd.
    For example, we may choose the eigenvalue_proportion = 0.9 which will add components to the reconstructed signal (from largest to smallest) until 90% of the psd is achieved. The remaining components (which make up the remaining 10%) will be classified as noise.
    Alternatively, we may choose eigenvalue_proportion = -0.1 which means only components that comtribute at least 10% to the psd will be classified as trend or periodic, the rest as noise.


    Parameters
    ----------
    Z : np.ndarray
        DESCRIPTION: Output CiSSA results (from the run_cissa function).
    psd : np.ndarray
        DESCRIPTION: estimation of the the circulant matrix power spectral density
    L : int
        DESCRIPTION: CiSSA window length.
    eigenvalue_proportion : TYPE
        DESCRIPTION: Option for how to select components to keep. There are three options, although the main purpose of this function is options 2 and 3:
                1) A dictionary. Each value contains a numpy row vector with the desired
                values of k to be included in a group, k=0,1,2,...,L/2-1. The function
                computes the reconstructed components for these groups.
                2) A number between 0 & 1. This number represents the accumulated
                share of the psd achieved with the sum of the share associated to
                the largest eigenvalues. The function computes the original
                reconstructed time series as the sum of these components.
                3) A number between -1 & 0. It is a percentile (in positive) of
                the psd. The function computes the original reconstructed time
                series as the sum of the reconstructed componentes by frequency
                whose psd is greater that this percentile.

    Returns
    -------
    trend : list
        DESCRIPTION: List of array locations for the trend component
    periodic : list
        DESCRIPTION: List of array locations for the periodic components
    noise : list
        DESCRIPTION: List of array locations for the noise components
    '''
    myfrequencies = generate_grouping(psd,L, trend=True)
    rc, sh, kg, _ = group(Z,psd,eigenvalue_proportion)

    trend = []
    if 0 in list(kg.values())[0]: trend.append(0)

    periodic = sorted([x for x in list(kg.values())[0] if x != 0])

    noise = sorted([x for x in range(0,max(myfrequencies.values())[0]+1) if x not in periodic and x not in trend])

    return trend,periodic,noise
###############################################################################
###############################################################################
def drop_monte_carlo_non_significant_components(Z,tempresults,surrogates,alpha):
    '''
    Function to remove components that do not pass the monte carlo test.

    Parameters
    ----------
    Z : np.ndarray
        DESCRIPTION: Output CiSSA results (from the run_cissa function).
    tempresults : dict
        DESCRIPTION: CiSSA results dictionary
    surrogates : str
        DESCRIPTION: Type of surrogates to fit. One of "random_permutation", "small_shuffle", "ar1_fit"
    alpha : float
        DESCRIPTION: Significance level for surrogate hypothesis test. For example, --> 100*(1-alpha)% confidence interval. The default is 0.05 (a 95% confidence interval).

    Returns
    -------
    new_array : np.ndarray
        DESCRIPTION: Reconstructed time-series with components that fail monte carlo test removed.

    '''
    new_array = np.zeros(Z[:,0:1].shape)
    for results_key_k in tempresults.get('components').keys():
        if tempresults.get('components').get(results_key_k).get('monte_carlo').get(surrogates).get('alpha').get(alpha).get('pass'):
            rc_length= len(tempresults.get('components').get(results_key_k).get('reconstructed_data'))
            new_array += tempresults.get('components').get(results_key_k).get('reconstructed_data').reshape(rc_length,1)
    return new_array

def classify_monte_carlo_non_significant_components(Z,tempresults):
    '''
    Function to classify components that do not pass the monte carlo test.

    Parameters
    ----------
    Z : np.ndarray
        DESCRIPTION: Output CiSSA results (from the run_cissa function).
    tempresults : dict
        DESCRIPTION: CiSSA results dictionary

    Returns
    -------
    trend : list
        DESCRIPTION: List of array locations for the trend component
    periodic : list
        DESCRIPTION: List of array locations for the periodic components
    noise : list
        DESCRIPTION: List of array locations for the noise components

    '''
    surrogate_type = tempresults['model parameters']['monte_carlo_surrogate_type']
    alpha = tempresults['model parameters']['monte_carlo_alpha']
    trend = []
    periodic = []
    noise = []

    for key_j in tempresults['components'].keys():
        mc_pass = tempresults['components'][key_j]['monte_carlo'][surrogate_type]['alpha'][alpha]['pass']
        if key_j == 'trend':
            trend.append(0)
        else:
            if mc_pass:
                periodic.append(tempresults['components'][key_j]['array_position'])
            else:
                noise.append(tempresults['components'][key_j]['array_position'])
    return trend,periodic,noise

###############################################################################
###############################################################################

###############################################################################
###############################################################################

###############################################################################
###############################################################################