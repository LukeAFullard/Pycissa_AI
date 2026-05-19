import numpy as np
import scipy.linalg
from pycissa.processing.matrix_operations.matrix_operations import diagonal_average_vectorized, calculate_number_of_frequencies

# TODO: Optimize with further vectorization after validation.

def create_m_trajectory_matrix(x_e: np.ndarray, L: int, M: int) -> np.ndarray:
    """
    Creates the multivariate trajectory matrix.
    x_e: shape (T_ext, M)
    """
    T_ext = x_e.shape[0]
    N = T_ext - L + 1
    LM = L * M
    X = np.zeros((LM, N))

    for j in range(L):
        # x_e[j : N + j, :] has shape (N, M)
        # Transpose to (M, N)
        X[j*M : (j+1)*M, :] = x_e[j : N + j, :].T

    return X

def create_m_autocovariance(x: np.ndarray, L: int, T: int, M: int) -> np.ndarray:
    """
    Cross-covariance matrix function.
    x: shape (T, M)
    Returns Gam: shape (M, M, L)
    """
    Gam = np.zeros((M, M, L))
    mean_x = np.mean(x, axis=0) # shape (M,)

    for m in range(L):
        x_1 = x[0:T-m, :] - mean_x
        x_2 = x[m:T, :] - mean_x
        Gam[:, :, m] = (x_1.T @ x_2) / (T - m)

    return Gam

def create_m_toeplitz_circulant(Gam: np.ndarray, L: int, M: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Block Toeplitz cross-covariance matrix S and equivalent block circulant matrix C
    Gam: shape (M, M, L)
    """
    S = np.kron(np.eye(L), Gam[:, :, 0])
    C = S.copy()

    for j in range(L):
        for k in range(j+1, L):
            m = np.abs(j - k)
            S[j*M : (j+1)*M, k*M : (k+1)*M] = Gam[:, :, m]
            S[k*M : (k+1)*M, j*M : (j+1)*M] = Gam[:, :, m].T

            C[j*M : (j+1)*M, k*M : (k+1)*M] = ((L - m) / L) * Gam[:, :, m] + (m / L) * Gam[:, :, L - m].T
            C[k*M : (k+1)*M, j*M : (j+1)*M] = ((L - m) / L) * Gam[:, :, m].T + (m / L) * Gam[:, :, L - m]

    return S, C

def m_cross_spectral_density_and_eigenvectors(C: np.ndarray, L: int, M: int) -> tuple[np.ndarray, list, np.ndarray]:
    """
    Diagonalization of cross spectral density matrices and computation of real eigenvectors.
    """
    U = scipy.linalg.dft(L) / np.sqrt(L)

    Fc = np.sqrt(L) * np.kron(U, np.eye(M)).conj().T @ C[:, 0:M]

    E = []
    D = []
    for k in range(L):
        F_k = Fc[k*M : (k+1)*M, :]
        # Diagonalize
        eigenvalues, eigenvectors = scipy.linalg.eig(F_k)

        # Sort descending by magnitude to match typical SSA / eigs
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        E.append(eigenvectors)
        D.append(np.abs(eigenvalues))

    V = np.kron(U, np.eye(M)) @ scipy.linalg.block_diag(*E)

    # Create a copy because we are updating V in place and indexing
    V_real = V.copy()
    V_real[:, 0:M] = np.real(V[:, 0:M])

    nf2, nft = calculate_number_of_frequencies(L)
    for k in range(1, int(nf2) + 1):
        v_l = V[:, k*M : (k+1)*M]
        V_real[:, k*M : (k+1)*M] = np.sqrt(2) * np.real(v_l)
        V_real[:, (L-k)*M : (L-k+1)*M] = np.sqrt(2) * np.imag(v_l)

    if L % 2 == 0:
        V_real[:, M*int(nft-1) : M*int(nft)] = np.real(V_real[:, M*int(nft-1) : M*int(nft)])

    V = np.real(V_real)

    return V, D, np.array(D)

def diagaver_m(Y: np.ndarray, M: int) -> np.ndarray:
    """
    Diagonal averaging for Multivariate SSA.
    Y: shape (LM, N)
    """
    LL, NN = Y.shape
    if LL % M != 0 or LL / M > NN:
        Y = Y.T
    LL, N = Y.shape
    L = LL // M
    T = N + L - 1
    y = np.zeros((T, M))

    for i in range(M):
        # Y[i::M, :] extracts the i-th component block, shape (L, N)
        y[:, i] = diagonal_average_vectorized(Y[i::M, :]).flatten()

    return y

def m_reconstruction(V: np.ndarray, W: np.ndarray, L: int, M: int, T_ext: int) -> tuple[list, list]:
    """
    Reconstructs elementary components and subcomponents.
    Returns Rs and R.
    Rs is a list of length M, each element is shape (T_ext, M, L).
    R is a list of length M, each element is shape (T_ext, L).
    """
    Rs = [np.zeros((T_ext, M, L)) for _ in range(M)]

    for k in range(L):
        for m in range(M):
            Y_km = np.outer(V[:, k*M + m], W[k*M + m, :])
            y = diagaver_m(Y_km, M)
            for j in range(M):
                Rs[j][:, m, k] = y[:, j]

    R = [np.sum(Rs[j], axis=1) for j in range(M)]
    return Rs, R

def m_group_paired_frequencies(Rs: list, R: list, L: int, M: int, T_ext: int, actual_left_ext: int, actual_right_ext: int) -> tuple[list, list]:
    """
    Groups the reconstructed subcomponents by frequency.
    Returns Zs (list of M arrays of shape (T, M, nft)) and Z (list of M arrays of shape (T, nft)).
    """
    nf2, nft = calculate_number_of_frequencies(L)

    Zs = [np.zeros((T_ext, M, int(nft))) for _ in range(M)]

    for j in range(M):
        Zs[j][:, :, 0] = Rs[j][:, :, 0]
        for k in range(1, int(nf2) + 1):
            Zs[j][:, :, k] = Rs[j][:, :, k] + Rs[j][:, :, L - k]

        if L % 2 == 0:
            Zs[j][:, :, int(nft - 1)] = Rs[j][:, :, int(nft - 1)]

        end_idx = T_ext - actual_right_ext
        if end_idx == T_ext:
            end_idx = None

        Zs[j] = Zs[j][actual_left_ext:end_idx, :, :]

    Z = [np.sum(Zs[j], axis=1) for j in range(M)]

    return Zs, Z

def run_mcissa(x: np.ndarray, L: int, extension_type: str = 'AR_LR', extend_left: bool = True, extend_right: bool = True) -> tuple[list, np.ndarray, list]:
    """
    Runs M-CiSSA algorithm.
    x: shape (T, M)
    """
    from pycissa.utilities.extendseries import extend_series
    from pycissa.processing.matrix_operations.matrix_operations import define_left_and_right_extension_lengths

    T, M = x.shape

    left_ext, right_ext = define_left_and_right_extension_lengths(extension_type, T, L)
    actual_left_ext = left_ext if extend_left else 0
    actual_right_ext = right_ext if extend_right else 0

    # Extend each variable independently
    x_e = np.zeros((T + actual_left_ext + actual_right_ext, M))
    for i in range(M):
        extended_col = extend_series(x[:, [i]], extension_type, left_ext, right_ext, extend_left=extend_left, extend_right=extend_right)
        x_e[:, i] = extended_col.flatten()

    T_ext = x_e.shape[0]

    X = create_m_trajectory_matrix(x_e, L, M)
    Gam = create_m_autocovariance(x, L, T, M)
    S, C = create_m_toeplitz_circulant(Gam, L, M)

    V, D_list, D_array = m_cross_spectral_density_and_eigenvectors(C, L, M)

    W = V.T @ X

    Rs, R = m_reconstruction(V, W, L, M, T_ext)

    Zs, Z = m_group_paired_frequencies(Rs, R, L, M, T_ext, actual_left_ext, actual_right_ext)

    # Returning Z as shape (T, M, nft) and psd (eigenvalues sum or concatenated)
    # The eigenvalues matrix for each freq is M x M. We typically need a 1D array of eigenvalues (psd) for compatibility.
    # We can provide a flattened array or keep it structured.
    # D_array has shape (L, M). We can flatten it or return as is.
    # Let's stack Z into (T, M, nft).
    Z_stacked = np.stack(Z, axis=1) # shape (T, M, nft)

    # Flatten D_array into psd shape (L * M, 1) or keep it (L, M)
    psd = D_array # shape (L, M)

    return Z_stacked, psd, Zs
