import numpy as np 
from scipy.optimize import fminbound
import itertools
from scipy.linalg import logm

def ket(dim, args):
    # Helper to decide whether an object is sequence-like (but not a string)
    is_seq = lambda obj: isinstance(obj, (list, tuple))
    # Convert inputs to lists for uniform processing
    dims = list(dim) if is_seq(dim) else [int(dim)]
    idxs = list(args) if is_seq(args) else [int(args)]
    # If a single dimension is provided for multiple indices, replicate it
    if len(dims) == 1 and len(idxs) > 1:
        dims *= len(idxs)
    if len(dims) != len(idxs):
        raise ValueError("Length mismatch between 'dim' and 'args'.")
    # Build individual basis vectors |j_k⟩ of length d_k
    basis_vectors = []
    for d_k, j_k in zip(dims, idxs):
        if not (0 <= j_k < d_k):
            raise ValueError(f"Index {j_k} out of range for dimension {d_k}.")
        v = [0.0] * d_k
        v[j_k] = 1.0
        basis_vectors.append(v)
    # Tensor (Kronecker) product of all subsystem basis vectors
    def kron(a, b):
        return [ai * bj for ai in a for bj in b]
    out = basis_vectors[0]
    for v in basis_vectors[1:]:
        out = kron(out, v)
    return out
def tensor(*args):
    '''
    Takes the tensor product of an arbitrary number of matrices/vectors.
    
    Input
    -----
    args : any number of nd arrays of floats
        Each argument is a matrix or vector (array-like) to include in the
        Kronecker product.
    
    Output
    ------
    M : numpy.ndarray
        The tensor (Kronecker) product of all input arrays.  If every input is
        one-dimensional the result is a 1-D array; otherwise it is at least
        two-dimensional.
    '''
    if len(args) == 0:
        raise ValueError("tensor() requires at least one input array.")
    M = np.asarray(args[0], dtype=float)
    for factor in args[1:]:
        M = np.kron(M, np.asarray(factor, dtype=float))
    return M
def apply_channel(K, rho, sys=None, dim=None):
    rho = np.asarray(rho, dtype=float)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("`rho` must be a square 2-D array (density matrix).")
    d_tot = rho.shape[0]
    def _tensor(*arrays):
        out = np.asarray(arrays[0], dtype=float)
        for arr in arrays[1:]:
            out = np.kron(out, np.asarray(arr, dtype=float))
        return out
    if sys is None or dim is None:
        out = np.zeros_like(rho, dtype=float)
        for Ki in K:
            Ki = np.asarray(Ki, dtype=float)
            if Ki.shape != (d_tot, d_tot):
                raise ValueError("Kraus operator dimension mismatch.")
            out += Ki @ rho @ Ki.T
        return out
    if not isinstance(sys, (list, tuple)) or len(sys) != 1:
        raise NotImplementedError("Only single-subsystem channels supported.")
    if not isinstance(dim, (list, tuple)):
        raise ValueError("`dim` must be a list/tuple when `sys` is provided.")
    if np.prod(dim) != d_tot:
        raise ValueError("Product of `dim` must equal the dimension of `rho`.")
    tgt = sys[0]
    if not (0 <= tgt < len(dim)):
        raise ValueError("Subsystem index out of range.")
    d_tgt = dim[tgt]
    identities = [np.eye(d, dtype=float) for d in dim]
    out = np.zeros((d_tot, d_tot), dtype=float)
    for Ki in K:
        Ki = np.asarray(Ki, dtype=float)
        if Ki.shape != (d_tgt, d_tgt):
            raise ValueError("Kraus operator incompatible with target dimension.")
        factors = identities.copy()
        factors[tgt] = Ki
        G = _tensor(*factors)
        out += G @ rho @ G.T
    return out
def syspermute(X, perm, dim):
    X = [list(map(float, row)) for row in X]
    if not X or any(len(row) != len(X) for row in X):
        raise ValueError("`X` must be a square 2-D array.")
    if not isinstance(dim, (list, tuple)) or not all(isinstance(d, int) and d > 0 for d in dim):
        raise ValueError("`dim` must be a list/tuple of positive integers.")
    if not isinstance(perm, (list, tuple)):
        raise ValueError("`perm` must be a list/tuple.")
    n = len(dim)
    if len(perm) != n or sorted(perm) != list(range(n)):
        raise ValueError("`perm` must be a permutation of range(len(dim)).")
    d_tot = 1
    for d in dim:
        d_tot *= d
    if len(X) != d_tot:
        raise ValueError("Dimension mismatch: prod(dim) must equal size of `X`.")
    def idx_to_digits(idx, bases):
        digits = [0] * len(bases)
        for k in range(len(bases) - 1, -1, -1):
            digits[k] = idx % bases[k]
            idx //= bases[k]
        return digits
    def digits_to_idx(digits, bases):
        idx = 0
        for d, base in zip(digits, bases):
            idx = idx * base + d
        return idx
    dim_perm = [dim[p] for p in perm]
    Y = [[0.0 for _ in range(d_tot)] for _ in range(d_tot)]
    for i in range(d_tot):
        digits_i = idx_to_digits(i, dim)
        perm_digits_i = [digits_i[p] for p in perm]
        new_i = digits_to_idx(perm_digits_i, dim_perm)
        row_X = X[i]
        row_Y = Y[new_i]
        for j in range(d_tot):
            digits_j = idx_to_digits(j, dim)
            perm_digits_j = [digits_j[p] for p in perm]
            new_j = digits_to_idx(perm_digits_j, dim_perm)
            row_Y[new_j] = row_X[j]
    return Y
def partial_trace(X, sys, dim):
    def _prod(iterable):
        p = 1
        for v in iterable:
            p *= v
        return p

    def _index_to_multi(idx, dims_list):
        if not dims_list:
            return []
        res = [0] * len(dims_list)
        for k in range(len(dims_list) - 1, -1, -1):
            res[k] = idx % dims_list[k]
            idx //= dims_list[k]
        return res

    def _multi_to_flat(indices, strides):
        return sum(i * s for i, s in zip(indices, strides))

    rho = [list(map(float, row)) for row in X]
    if not rho or any(len(row) != len(rho) for row in rho):
        raise ValueError("`X` must be a square 2-D array.")
    total_dim = len(rho)

    if not isinstance(dim, (list, tuple)) or not all(isinstance(d, int) and d > 0 for d in dim):
        raise ValueError("`dim` must be a list/tuple of positive integers.")
    if _prod(dim) != total_dim:
        raise ValueError("Product of `dim` must equal the dimension of `X`.")

    n = len(dim)
    trace_sys = sorted(sys or [])
    if not isinstance(trace_sys, list):
        raise ValueError("`sys` must be a list or tuple.")
    if len(set(trace_sys)) != len(trace_sys):
        raise ValueError("`sys` contains duplicate indices.")
    if any(not (0 <= s < n) for s in trace_sys):
        raise ValueError("Subsystem index out of range.")

    if not trace_sys:
        return rho

    keep_sys = [k for k in range(n) if k not in trace_sys]
    dims_keep = [dim[k] for k in keep_sys]
    dims_tr = [dim[k] for k in trace_sys]
    d_keep = _prod(dims_keep) if dims_keep else 1
    d_tr = _prod(dims_tr) if dims_tr else 1

    strides = []
    acc = 1
    for d in reversed(dim):
        strides.insert(0, acc)
        acc *= d

    keep_pos_map = {pos: i for i, pos in enumerate(keep_sys)}
    tr_pos_map = {pos: i for i, pos in enumerate(trace_sys)}

    reduced = [[0.0 for _ in range(d_keep)] for _ in range(d_keep)]

    for i_keep in range(d_keep):
        idx_keep_row = _index_to_multi(i_keep, dims_keep) if keep_sys else []
        for j_keep in range(d_keep):
            idx_keep_col = _index_to_multi(j_keep, dims_keep) if keep_sys else []
            s = 0.0
            for i_tr in range(d_tr):
                idx_tr = _index_to_multi(i_tr, dims_tr) if trace_sys else []
                full_row = [0] * n
                full_col = [0] * n
                for pos in range(n):
                    if pos in keep_pos_map:
                        k_idx = keep_pos_map[pos]
                        full_row[pos] = idx_keep_row[k_idx]
                        full_col[pos] = idx_keep_col[k_idx]
                    else:
                        t_idx = tr_pos_map[pos]
                        full_row[pos] = idx_tr[t_idx]
                        full_col[pos] = idx_tr[t_idx]
                flat_row = _multi_to_flat(full_row, strides)
                flat_col = _multi_to_flat(full_col, strides)
                s += rho[flat_row][flat_col]
            reduced[i_keep][j_keep] = s

    return reduced
def entropy(rho):
    rho = np.asarray(rho, dtype=float)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("`rho` must be a square 2-D array.")
    if not np.allclose(rho, rho.T):
        raise ValueError("`rho` must be Hermitian.")
    eigvals = np.linalg.eigvalsh(rho)
    tol = 1e-12
    eigvals[eigvals < tol] = 0.0
    pos = eigvals > 0.0
    if not np.any(pos):
        return 0.0
    en = -np.sum(eigvals[pos] * np.log(eigvals[pos]))
    return float(max(en, 0.0))
def generalized_amplitude_damping_channel(gamma, N):
    '''Generates the generalized amplitude damping channel.
    Inputs:
    gamma: float, damping parameter (0 ≤ gamma ≤ 1)
    N: float,   thermal parameter (0 ≤ N ≤ 1)

    Output:
    kraus: list of four 2×2 lists (dtype=float) representing the Kraus
           operators [K1, K2, K3, K4] of the channel 𝒜_{γ,N}.
    '''
    try:
        gamma = float(gamma)
        N = float(N)
    except (TypeError, ValueError):
        raise ValueError("`gamma` and `N` must be real numbers.")
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("`gamma` must lie in the interval [0, 1].")
    if not (0.0 <= N <= 1.0):
        raise ValueError("`N` must lie in the interval [0, 1].")

    sqrt_1_minus_N   = (1.0 - N) ** 0.5
    sqrt_N           = N ** 0.5
    sqrt_1_minus_g   = (1.0 - gamma) ** 0.5
    sqrt_gamma_1mN   = (gamma * (1.0 - N)) ** 0.5
    sqrt_gamma_N     = (gamma * N) ** 0.5

    K1 = [[1.0 * sqrt_1_minus_N, 0.0],
          [0.0, sqrt_1_minus_N * sqrt_1_minus_g]]

    K2 = [[0.0,             sqrt_gamma_1mN],
          [0.0,             0.0           ]]

    K3 = [[sqrt_N * sqrt_1_minus_g, 0.0],
          [0.0,                    sqrt_N * 1.0]]

    K4 = [[0.0,        0.0],
          [sqrt_gamma_N, 0.0]]

    return [K1, K2, K3, K4]
def neg_rev_coh_info(p, g, N):
    """
    Calculates the negative of reverse coherent information of the output
    state obtained by sending the second qubit of
        |ψ⟩ = √(1 − p)|00⟩ + √p |11⟩
    through a generalized-amplitude-damping channel 𝒜_{g,N}.

    Parameters
    ----------
    p : float
        Amplitude parameter of the input state (0 ≤ p ≤ 1).
    g : float
        Damping parameter γ of the GADC (0 ≤ g ≤ 1).
    N : float
        Thermal parameter N of the GADC (0 ≤ N ≤ 1).

    Returns
    -------
    float
        The negative reverse coherent information:
            S(ρ_out) − S(ρ_A),
        where ρ_out is the two-qubit state after the channel and ρ_A is the
        reduced state of the qubit that did not go through the channel.
    """
    try:
        p, g, N = float(p), float(g), float(N)
    except (TypeError, ValueError):
        raise ValueError('`p`, `g`, and `N` must be real numbers.')
    for name, val in (('p', p), ('g', g), ('N', N)):
        if not (0.0 <= val <= 1.0):
            raise ValueError(f'`{name}` must lie in the interval [0, 1].')
    psi = np.array([np.sqrt(1.0 - p), 0.0, 0.0, np.sqrt(p)], dtype=float)
    rho_in = np.outer(psi, psi)
    kraus_ops = generalized_amplitude_damping_channel(gamma=g, N=N)
    rho_out = apply_channel(kraus_ops, rho_in, sys=[1], dim=[2, 2])
    rho_A = partial_trace(rho_out.tolist(), sys=[1], dim=[2, 2])
    S_AB = entropy(rho_out)
    S_A = entropy(rho_A)
    neg_I_R = S_AB - S_A
    return float(neg_I_R)
def GADC_rev_coh_inf(g, N):
    '''Calculates the channel reverse coherent information of a GADC.
    
    Parameters
    ----------
    g : float
        Damping parameter γ of the generalized-amplitude-damping channel
        (0 ≤ g ≤ 1).
    N : float
        Thermal parameter of the channel (0 ≤ N ≤ 1).

    Returns
    -------
    channel_coh_info : float
        The channel reverse coherent information
            I_R(𝒜_{g,N}) = max_{0≤p≤1} I_R(A⟩B)_{ρ(p)},
        where ρ(p) is obtained from the state
            |ψ(p)⟩ = √(1−p)|00⟩ + √p|11⟩
        after the channel acts on the second qubit.
    '''
    # ----------------- input validation -----------------
    try:
        g = float(g)
        N = float(N)
    except (TypeError, ValueError):
        raise ValueError('`g` and `N` must be real numbers.')
    for name, val in (('g', g), ('N', N)):
        if not (0.0 <= val <= 1.0):
            raise ValueError(f'`{name}` must lie in the interval [0, 1].')

    # --------------- optimisation over p ∈ [0,1] ---------------
    def objective(p):
        # minimise the negative ⇒ maximise the positive
        return neg_rev_coh_info(p, g, N)

    # fminbound finds minimum over open interval; include endpoints manually
    # from scipy.optimize import fminbound
    _, f_min, *_ = fminbound(objective, 0.0, 1.0, full_output=True, disp=0)
    f_min = min(f_min, objective(0.0), objective(1.0))

    # Channel reverse coherent information
    channel_coh_info = -f_min

    # Remove potential tiny negative artefacts
    if channel_coh_info < 0 and abs(channel_coh_info) < 1e-12:
        channel_coh_info = 0.0

    return float(channel_coh_info)

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('71.9', 5)
target = targets[0]

assert np.allclose(GADC_rev_coh_inf(0.2,0.4), target)
target = targets[1]

assert np.allclose(GADC_rev_coh_inf(0.2,0.1), target)
target = targets[2]

assert np.allclose(GADC_rev_coh_inf(0.4,0.2), target)
target = targets[3]

assert np.allclose(GADC_rev_coh_inf(0,0), target)
target = targets[4]

assert np.allclose(GADC_rev_coh_inf(1,1), target)
