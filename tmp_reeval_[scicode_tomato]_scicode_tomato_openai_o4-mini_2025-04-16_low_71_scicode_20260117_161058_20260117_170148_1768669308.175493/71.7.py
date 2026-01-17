import numpy as np 
from scipy.optimize import fminbound
import itertools
from scipy.linalg import logm

def ket(dim, args):
    """
    Input:
        dim: int or list of ints
            Dimension(s) of the ket space.
        args: int or list of ints
            Index (or indices) of the basis vector(s).
    Output:
        out: numpy.ndarray
            The resulting ket as a one-hot vector (or tensor product thereof).
    """
    if isinstance(args, int):
        arg_list = [args]
    else:
        arg_list = list(args)
    if isinstance(dim, int):
        dim_list = [dim] * len(arg_list)
    else:
        dim_list = list(dim)
    if len(dim_list) != len(arg_list):
        raise ValueError(f"Length mismatch: dim has {len(dim_list)} elements, "
                         f"args has {len(arg_list)} elements.")
    vecs = []
    for d_i, j_i in zip(dim_list, arg_list):
        if not (0 <= j_i < d_i):
            raise ValueError(f"Index {j_i} out of range for dimension {d_i}.")
        v = np.zeros(d_i, dtype=float)
        v[j_i] = 1.0
        vecs.append(v)
    result = vecs[0]
    for v in vecs[1:]:
        result = np.kron(result, v)
    return result
def tensor(*args):
    '''Takes the tensor product of an arbitrary number of matrices/vectors.
    Input:
        args: any number of nd arrays of floats, corresponding to input matrices
    Output:
        M: the tensor product (Kronecker product) of input matrices/vectors, as an nd array of floats
    '''
    if not args:
        raise ValueError("tensor() requires at least one input array.")
    result = np.array(args[0], dtype=float)
    for arr in args[1:]:
        current = np.array(arr, dtype=float)
        result = np.kron(result, current)
    return result
def apply_channel(K, rho, sys=None, dim=None):
    if not isinstance(K, (list, tuple)) or len(K) == 0:
        raise ValueError("K must be a non-empty list of Kraus operators.")
    for Ki in K:
        if not (isinstance(Ki, np.ndarray) and Ki.ndim == 2 and Ki.shape[0] == Ki.shape[1]):
            raise ValueError("Each Kraus operator must be a square 2D numpy array.")
    if not (isinstance(rho, np.ndarray) and rho.ndim == 2 and rho.shape[0] == rho.shape[1]):
        raise ValueError("rho must be a square 2D numpy array.")
    if sys is None and dim is None:
        out = np.zeros_like(rho, dtype=complex)
        for Ki in K:
            out += np.dot(Ki, np.dot(rho, Ki.conj().T))
        return out.real if np.isrealobj(out) else out
    if sys is None or dim is None:
        raise ValueError("Either both sys and dim must be None, or both provided.")
    sys_list = list(sys) if isinstance(sys, (list, tuple)) else [int(sys)]
    dim_list = list(dim) if isinstance(dim, (list, tuple)) else [int(dim)]
    n_sub = len(dim_list)
    for s in sys_list:
        if not (0 <= s < n_sub):
            raise ValueError(f"Subsystem index {s} out of range for {n_sub} subsystems.")
    total_dim = int(np.prod(dim_list))
    if rho.shape != (total_dim, total_dim):
        raise ValueError(f"rho.shape {rho.shape} does not match total dimension {total_dim}.")
    id_list = [np.eye(d, dtype=complex) for d in dim_list]
    out = np.zeros((total_dim, total_dim), dtype=complex)
    for idx_tuple in itertools.product(range(len(K)), repeat=len(sys_list)):
        ops = []
        for sub in range(n_sub):
            if sub in sys_list:
                pos = sys_list.index(sub)
                ops.append(K[idx_tuple[pos]])
            else:
                ops.append(id_list[sub])
        K_global = ops[0]
        for M in ops[1:]:
            K_global = np.kron(K_global, M)
        out += np.dot(K_global, np.dot(rho, K_global.conj().T))
    return out.real if np.isrealobj(out) else out
def syspermute(X, perm, dim):
    '''Permutes order of subsystems in the multipartite operator X.
    Inputs:
        X: 2d square numpy array, the density matrix of the state
        perm: list of int containing the desired order
        dim: list of int containing the dimensions of all subsystems.
    Output:
        Y: 2d square numpy array, the density matrix of the permuted state
    '''
    if not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[0] != X.shape[1]:
        raise ValueError("X must be a square 2D numpy array.")
    if not (isinstance(dim, (list, tuple)) and all(isinstance(d, int) and d > 0 for d in dim)):
        raise ValueError("dim must be a list of positive integers.")
    n = len(dim)
    total_dim = int(np.prod(dim))
    if X.shape != (total_dim, total_dim):
        raise ValueError(f"X.shape {X.shape} does not match product of dim {total_dim}.")
    if not (isinstance(perm, (list, tuple)) and len(perm) == n and sorted(perm) == list(range(n))):
        raise ValueError("perm must be a permutation of 0..len(dim)-1.")
    tensor_shape = tuple(dim) + tuple(dim)
    X_tensor = X.reshape(tensor_shape)
    axes = list(perm) + [p + n for p in perm]
    X_permuted = X_tensor.transpose(axes)
    Y = X_permuted.reshape((total_dim, total_dim))
    return Y
def partial_trace(X, sys, dim):
    if not (isinstance(X, np.ndarray) and X.ndim == 2 and X.shape[0] == X.shape[1]):
        raise ValueError("X must be a square 2D numpy array.")
    if not (isinstance(dim, (list, tuple)) and all(isinstance(d, int) and d > 0 for d in dim)):
        raise ValueError("dim must be a list of positive integers.")
    n = len(dim)
    total_dim = int(np.prod(dim))
    if X.shape[0] != total_dim:
        raise ValueError(f"X.shape {X.shape} does not match product of dim {total_dim}.")
    if not isinstance(sys, (list, tuple)):
        raise ValueError("sys must be a list or tuple of subsystem indices to trace out.")
    sys_list = list(sys)
    if len(set(sys_list)) != len(sys_list):
        raise ValueError("sys contains duplicate indices.")
    for s in sys_list:
        if not (0 <= s < n):
            raise ValueError(f"Subsystem index {s} out of range [0, {n}).")
    keep = [i for i in range(n) if i not in sys_list]
    trace = sys_list
    perm = keep + trace
    Xp = syspermute(X, perm, dim)
    dA = int(np.prod([dim[i] for i in keep])) if keep else 1
    dB = int(np.prod([dim[i] for i in trace])) if trace else 1
    X4 = Xp.reshape((dA, dB, dA, dB))
    return np.trace(X4, axis1=1, axis2=3)
def entropy(rho):
    """
    Inputs:
        rho: 2d numpy array (square), the density matrix of the state
    Output:
        en: von Neumann entropy of rho (float)
    """
    if not isinstance(rho, np.ndarray) or rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho must be a square 2D numpy array.")
    vals = np.linalg.eigvalsh(rho)
    tol = 1e-12
    vals = vals[vals > tol]
    en = -np.sum(vals * np.log(vals))
    return float(en)
def generalized_amplitude_damping_channel(gamma, N):
    """
    Generates the generalized amplitude damping channel.
    Inputs:
        gamma: float, damping parameter
        N: float, thermal parameter
    Output:
        kraus: list of Kraus operators as 2x2 arrays of floats [K1, K2, K3, K4]
    """
    a = np.sqrt(1.0 - gamma)
    b = np.sqrt(gamma)
    p = np.sqrt(1.0 - N)
    q = np.sqrt(N)
    P00 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=float)
    P11 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=float)
    P01 = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=float)
    P10 = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    K1 = p * (P00 + a * P11)
    K2 = b * p * P01
    K3 = q * (a * P00 + P11)
    K4 = b * q * P10
    return [K1, K2, K3, K4]

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('71.7', 3)
target = targets[0]

assert np.allclose(generalized_amplitude_damping_channel(0, 0), target)
target = targets[1]

assert np.allclose(generalized_amplitude_damping_channel(0.8, 0), target)
target = targets[2]

assert np.allclose(generalized_amplitude_damping_channel(0.5, 0.5), target)
