import numpy as np
import copy
import matplotlib.pyplot as plt

def m_get_surrogate_data(data: np.ndarray, surrogates: str, seed: int = None) -> np.ndarray:
    x_copy = copy.deepcopy(data)
    if surrogates == 'random_permutation':
        rng = np.random.default_rng(seed)
        # Apply the same permutation to all variables to maintain spatial cross-correlation
        indices = np.arange(x_copy.shape[0])
        rng.shuffle(indices)
        x_copy = x_copy[indices, :]
    elif surrogates == 'small_shuffle':
        # Apply same small shuffle to all variables
        rng = np.random.default_rng(seed)
        gaussian_random_numbers = rng.standard_normal(len(x_copy))
        original_index = np.arange(len(x_copy))
        perturbed_index = original_index + 1.0 * gaussian_random_numbers
        new_index = np.argsort(perturbed_index)
        x_copy = x_copy[new_index, :]
    else:
        # ar1_fit or others could be implemented per column, but let's stick to these for now.
        rng = np.random.default_rng(seed)
        indices = np.arange(x_copy.shape[0])
        rng.shuffle(indices)
        x_copy = x_copy[indices, :]
    return x_copy

def m_calculate_surrogate_psd(x_surrogate: np.ndarray, L: int, extension_type: str, extend_left: bool, extend_right: bool, nft: int) -> np.ndarray:
    from pycissa.processing.matrix_operations.m_matrix_operations import run_mcissa_psd_step
    psd_surrogate = run_mcissa_psd_step(x_surrogate, L=L, extension_type=extension_type, extend_left=extend_left, extend_right=extend_right)
    # Convert psd (L, M) to grouped frequency psd as in m_group
    psd_length = psd_surrogate.shape[0]

    if np.mod(psd_length, 2):
        pzz_surrogate = np.append(np.sum(psd_surrogate[0, :]), 2 * np.sum(psd_surrogate[1:nft, :], axis=1))
    else:
        pzz_surrogate = np.append(
            np.append(np.sum(psd_surrogate[0, :]), 2 * np.sum(psd_surrogate[1:nft-1, :], axis=1)),
            np.sum(psd_surrogate[nft-1, :])
        )
    return pzz_surrogate

def run_m_monte_carlo_test(x: np.ndarray,
                           L: int,
                           psd: np.ndarray,
                           results: dict,
                           alpha: float = 0.05,
                           K_surrogates: int = 1,
                           surrogates: str = 'random_permutation',
                           seed: int|None = None,
                           sided_test: str = 'one sided',
                           remove_trend: bool = True,
                           trend_always_significant: bool = True,
                           extension_type: str = 'AR_LR',
                           extend_left: bool = True,
                           extend_right: bool = True,
                           plot_figure: bool = False
                           ) -> tuple[dict, plt.figure]:
    from pycissa.postprocessing.monte_carlo.montecarlo import check_for_significance
    from pycissa.processing.matrix_operations.matrix_operations import calculate_number_of_frequencies

    if sided_test == 'one sided':
        number_of_surrogates = int(K_surrogates/alpha - 1)
    elif sided_test == 'two sided':
        number_of_surrogates = int(2*K_surrogates/alpha - 1)
    else:raise ValueError(f"The parameter sided_test must be one of 'one sided' or 'two sided'. You entered '{sided_test}'")

    result = copy.deepcopy(results)
    x_copy = copy.deepcopy(x)
    if remove_trend:
        # x_trend has shape (T, M)
        x_copy -= result.get('components').get('trend').get('reconstructed_data').reshape(x_copy.shape)

    nf2, nft = calculate_number_of_frequencies(L)
    psd_length = psd.shape[0]

    if np.mod(psd_length, 2):
        pzz = np.append(np.sum(psd[0, :]), 2 * np.sum(psd[1:int(nft), :], axis=1))
    else:
        pzz = np.append(
            np.append(np.sum(psd[0, :]), 2 * np.sum(psd[1:int(nft)-1, :], axis=1)),
            np.sum(psd[int(nft)-1, :])
        )

    surrogate_results = {}
    for surrogate_i in range(0, number_of_surrogates):
        current_seed = seed + surrogate_i if seed is not None else None
        x_surrogate = m_get_surrogate_data(x_copy, surrogates, current_seed)

        if remove_trend:
            x_surrogate += result.get('components').get('trend').get('reconstructed_data').reshape(x_surrogate.shape)

        pzz_surrogate = m_calculate_surrogate_psd(x_surrogate, L, extension_type, extend_left, extend_right, int(nft))

        for results_key_j in result.get('components').keys():
            key_array_position = result.get('components').get(results_key_j).get('array_position')
            surrogate_results.setdefault(results_key_j, []).append(pzz_surrogate[key_array_position])

    result, plot_period, surrogate_psd, signal_psd = check_for_significance(result, K_surrogates, alpha, pzz, surrogate_results, surrogates, trend_always_significant, sided_test)

    return result, None
