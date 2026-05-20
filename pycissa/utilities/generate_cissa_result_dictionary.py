import numpy as np

###############################################################################
###############################################################################

###############################################################################
###############################################################################

###############################################################################
###############################################################################

###############################################################################
###############################################################################

###############################################################################
###############################################################################

###############################################################################
###############################################################################
def generate_results_dictionary(Z:          np.ndarray,
                                psd:        np.ndarray,
                                L:          int,
                                cissa_type: str='cissa',
                                ):
    from pycissa.postprocessing.grouping.grouping_functions import generate_grouping, group

    myfrequencies = generate_grouping(psd,L, trend=True)
    rc, sh, kg, psd_sh = group(Z,psd,myfrequencies)

    results = {'components' : {}}
    for frequency_i in myfrequencies:
        if frequency_i == 'trend':
            results.get('components').update({frequency_i                 : {
                        'unitless period (number of timesteps)' : 0,
                        # 'reconstructed_data'                    : rc[frequency_i],
                        'reconstructed_data'                    : rc[frequency_i].reshape(len(rc[frequency_i]),),
                        'percentage_share_of_psd'               : sh[frequency_i],
                        # 'array_position'                        : kg[frequency_i],
                        'array_position'                        : kg[frequency_i][0],
                                                }
                })
        else:
            results.get('components').update({frequency_i                 : {
                        'unitless period (number of timesteps)' : 1/frequency_i,
                        # 'reconstructed_data'                    : rc[frequency_i],
                        'reconstructed_data'                    : rc[frequency_i].reshape(len(rc[frequency_i]),),
                        'percentage_share_of_psd'               : sh[frequency_i],
                        # 'array_position'                        : kg[frequency_i],
                        'array_position'                        : kg[frequency_i][0],
                                                }
                })
    return {cissa_type:results}

def generate_m_results_dictionary(Z_stacked: np.ndarray, psd: np.ndarray, L: int, cissa_type: str='mcissa'):
    from pycissa.postprocessing.grouping.grouping_functions import generate_grouping
    from pycissa.postprocessing.grouping.m_grouping_functions import m_group

    myfrequencies = generate_grouping(np.zeros(L), L, trend=True)
    rc, sh, kg, psd_sh = m_group(Z_stacked, psd, myfrequencies)

    results = {'components': {}}
    for frequency_i in myfrequencies:
        if frequency_i == 'trend':
            results.get('components').update({frequency_i: {
                'unitless period (number of timesteps)': 0,
                'reconstructed_data': rc[frequency_i],
                'percentage_share_of_psd': sh[frequency_i],
                'array_position': kg[frequency_i][0],
            }})
        else:
            results.get('components').update({frequency_i: {
                'unitless period (number of timesteps)': 1/frequency_i,
                'reconstructed_data': rc[frequency_i],
                'percentage_share_of_psd': sh[frequency_i],
                'array_position': kg[frequency_i][0],
            }})
    return {cissa_type: results}
