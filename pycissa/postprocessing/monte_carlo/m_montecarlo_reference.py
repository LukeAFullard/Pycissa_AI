import numpy as np
import copy
from pycissa.postprocessing.monte_carlo.m_montecarlo import m_get_surrogate_data
from pycissa.postprocessing.monte_carlo.montecarlo import check_for_significance
from pycissa.processing.matrix_operations.matrix_operations import calculate_number_of_frequencies

def m_calculate_reference_surrogate_psd(x_surrogate: np.ndarray, L: int, reference_indices: list, extension_type: str, extend_left: bool, extend_right: bool, nft: int) -> np.ndarray:
    from pycissa.processing.matrix_operations.m_matrix_operations import run_mcissa_psd_step
    # We only care about the power/variance in the reference channels.
    psd_surrogate = run_mcissa_psd_step(x_surrogate, L=L, extension_type=extension_type, extend_left=extend_left, extend_right=extend_right)

    psd_length = psd_surrogate.shape[0]

    if np.mod(psd_length, 2):
        pzz_surrogate = np.append(np.sum(psd_surrogate[0, reference_indices]), 2 * np.sum(psd_surrogate[1:nft, reference_indices], axis=1))
    else:
        pzz_surrogate = np.append(
            np.append(np.sum(psd_surrogate[0, reference_indices]), 2 * np.sum(psd_surrogate[1:nft-1, reference_indices], axis=1)),
            np.sum(psd_surrogate[nft-1, reference_indices])
        )
    return pzz_surrogate

def run_m_monte_carlo_reference_test(x: np.ndarray,
                           L: int,
                           psd: np.ndarray,
                           results: dict,
                           reference_indices: list,
                           alpha: float = 0.05,
                           K_surrogates: int = 1,
                           surrogates: str = 'random_permutation',
                           seed: int|None = None,
                           sided_test: str = 'one sided',
                           remove_trend: bool = True,
                           trend_always_significant: bool = True,
                           extension_type: str = 'AR_LR',
                           extend_left: bool = True,
                           extend_right: bool = True
                           ) -> dict:

    if alpha >= 1.0:
        number_of_surrogates = 0
    else:
        if sided_test == 'one sided':
            number_of_surrogates = int(K_surrogates/alpha - 1)
        elif sided_test == 'two sided':
            number_of_surrogates = int(2*K_surrogates/alpha - 1)
        else:raise ValueError(f"The parameter sided_test must be one of 'one sided' or 'two sided'. You entered '{sided_test}'")

    result = copy.deepcopy(results)

    # If alpha >= 1.0, we can short circuit and mark all as passing
    if alpha >= 1.0:
        for results_key_j in result.get('components').keys():
            result['components'][results_key_j].setdefault('monte_carlo', {})
            result['components'][results_key_j]['monte_carlo'].setdefault(surrogates, {})
            result['components'][results_key_j]['monte_carlo'][surrogates].setdefault('alpha', {})
            result['components'][results_key_j]['monte_carlo'][surrogates]['alpha'][alpha] = {'pass': True}
        return result
    x_copy = copy.deepcopy(x)
    if remove_trend:
        x_copy -= result.get('components').get('trend').get('reconstructed_data').reshape(x_copy.shape)

    nf2, nft = calculate_number_of_frequencies(L)
    psd_length = psd.shape[0]

    # Calculate empirical PSD only on the reference channels
    if np.mod(psd_length, 2):
        pzz = np.append(np.sum(psd[0, reference_indices]), 2 * np.sum(psd[1:int(nft), reference_indices], axis=1))
    else:
        pzz = np.append(
            np.append(np.sum(psd[0, reference_indices]), 2 * np.sum(psd[1:int(nft)-1, reference_indices], axis=1)),
            np.sum(psd[int(nft)-1, reference_indices])
        )

    surrogate_results = {}
    for surrogate_i in range(0, number_of_surrogates):
        current_seed = seed + surrogate_i if seed is not None else None
        x_surrogate = m_get_surrogate_data(x_copy, surrogates, current_seed)

        if remove_trend:
            x_surrogate += result.get('components').get('trend').get('reconstructed_data').reshape(x_surrogate.shape)

        pzz_surrogate = m_calculate_reference_surrogate_psd(x_surrogate, L, reference_indices, extension_type, extend_left, extend_right, int(nft))

        for results_key_j in result.get('components').keys():
            key_array_position = result.get('components').get(results_key_j).get('array_position')
            surrogate_results.setdefault(results_key_j, []).append(pzz_surrogate[key_array_position])

    result, plot_period, surrogate_psd, signal_psd = check_for_significance(result, K_surrogates, alpha, pzz, surrogate_results, surrogates, trend_always_significant, sided_test)

    return result
