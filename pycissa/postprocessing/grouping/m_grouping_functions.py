import numpy as np
from pycissa.postprocessing.grouping.grouping_functions import generate_grouping, find_smallest_values

def m_group(Z_stacked, psd, I, season_length=1, cycle_length=[1.5, 8], include_noise=True):
    """
    Multivariate extension of group.
    Z_stacked: shape (T, M, nft)
    psd: shape (L, M), eigenvalue array
    I: grouping option (int, dict, float)

    Returns:
    rc: dict of reconstructed components, each value has shape (T, M)
    sh: dict of percentage share of psd for each group
    kg: dict of indices belonging to each group
    psd_sh: dict of total psd sum for each group
    """
    T, M, nft = Z_stacked.shape
    L = psd.shape[0]

    # Calculate 1D psd representing total power per frequency
    # Note: sum of eigenvalues at each frequency gives the trace of cross-spectral density matrix,
    # which is the total power across all variables at that frequency.
    # We do the same frequency combination as univariate.
    if np.mod(L, 2):
        pzz = np.append(np.sum(psd[0, :]), 2 * np.sum(psd[1:nft, :], axis=1))
    else:
        pzz = np.append(
            np.append(np.sum(psd[0, :]), 2 * np.sum(psd[1:nft-1, :], axis=1)),
            np.sum(psd[nft-1, :])
        )

    if type(I) is dict:
        opc = 2
    elif type(I) == int or type(I) == float:
        if ((I - np.floor(I)) == 0) & (I > 0):
            opc = 1
        elif (0 < I) & (I < 1):
            opc = 3
        elif (-1 < I) & (I < 0):
            opc = 4
        else:
            raise ValueError(f'*** Input argument #3 (I): Value ({I}) not valid ***')
    else:
        raise ValueError(f'*** Input argument #3 ({I}): Type ({type(I)}) not valid ***')

    if opc == 1:
        if np.mod(L, I):
            raise ValueError(f'*** L is not proportional to the number of data per year (modulo of L/I = {np.mod(L,I)}) ***')
        G = 3
        if include_noise:
            G = 4
        s = I
        kg = {}
        kg.update({'seasonality': L * np.arange(1, np.floor(s/2)+1) / (season_length * s)})
        kg.update({'long term cycle': np.arange(max(1, np.floor(L/(cycle_length[1]*s)+1))-1, min(nft-1, np.floor(L/(cycle_length[0]*s)+1)), dtype=int)})
        kg.update({'trend': np.arange(0, kg['long term cycle'][0])})

        if include_noise:
            current_k = []
            for index_j in kg.values():
                current_k = current_k + [int(x) for x in index_j]
            missing_k = [x for x in range(0, int(np.floor(L/2))) if x not in current_k]
            kg.update({'noise': np.array(missing_k)})

    elif opc == 2:
        kg = I.copy()
    elif opc == 3:
        psor = np.sort(pzz)[::-1]
        ks = np.argsort(-pzz)
        pcum = 100 * np.cumsum(psor) / sum(pzz)
        kg = {1: ks[np.arange(0, len(ks[pcum < 100 * I]) + 1)]}
    elif opc == 4:
        ks = np.arange(0, nft)
        kg = {1: ks[pzz > np.percentile(pzz, -100 * I)]}

    rc = {}
    sh = {}
    psd_sh = {}
    total_pzz = np.sum(pzz)

    for key_j in kg.keys():
        indx = [int(x) for x in kg[key_j]]
        # Sum along the frequency axis
        rc.update({key_j: np.sum(Z_stacked[:, :, indx], axis=2)})
        sh.update({key_j: 100 * np.sum(pzz[indx]) / total_pzz})
        psd_sh.update({key_j: np.sum(pzz[indx])})

    return rc, sh, kg, psd_sh

def m_classify_smallest_proportion_psd(Z_stacked, psd, L, eigenvalue_proportion):
    myfrequencies = generate_grouping(np.zeros(L), L, trend=True)

    rc, sh, kg, _ = m_group(Z_stacked, psd, eigenvalue_proportion)
    trend = []
    if 0 in list(kg.values())[0]: trend.append(0)
    periodic = sorted([x for x in list(kg.values())[0] if x != 0])
    noise = sorted([x for x in range(0, max(myfrequencies.values())[0]+1) if x not in periodic and x not in trend])
    return trend, periodic, noise

def m_classify_smallest_n_components(Z_stacked, psd, L, number_of_groups_to_drop, include_trend=True):
    myfrequencies = generate_grouping(np.zeros(L), L, trend=include_trend)
    rc, sh, kg, psd_sh = m_group(Z_stacked, psd, myfrequencies)
    smallest_keys = find_smallest_values(sh, number_of_groups_to_drop)

    trend = []
    if 'trend' in kg:
        trend = [kg['trend'][0]]

    periodic = sorted([kg[x][0] for x in rc.keys() if x not in smallest_keys and x != 'trend'])
    noise = sorted([kg[x][0] for x in rc.keys() if x in smallest_keys])
    return trend, periodic, noise

def m_classify_monte_carlo_non_significant_components(tempresults):
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
    return trend, periodic, noise
