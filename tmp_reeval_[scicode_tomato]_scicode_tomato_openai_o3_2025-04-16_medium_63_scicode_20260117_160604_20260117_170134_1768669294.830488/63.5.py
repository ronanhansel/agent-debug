import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def initialize_grid(price_step, time_step, strike, max_price, min_price):
    if price_step < 2 or time_step < 2:
        raise ValueError("price_step and time_step must each be at least 2.")
    if min_price <= 0 or max_price <= min_price:
        raise ValueError("Require 0 < min_price < max_price.")
    def _linspace(start, stop, num):
        if num == 1:
            return [stop]
        step = (stop - start) / (num - 1)
        return [start + i * step for i in range(num)]
    p = _linspace(min_price, max_price, price_step)
    dp = (max_price - min_price) / (price_step - 1)
    T = _linspace(0.0, 1.0, time_step)
    dt = 1.0 / (time_step - 1)
    return p, dp, T, dt
def apply_boundary_conditions(N_p, N_t, p, T, strike, r, sig):
    if len(p) != N_p or len(T) != N_t:
        raise ValueError("Lengths of p and T must match N_p and N_t respectively.")
    if N_p < 2 or N_t < 2:
        raise ValueError("N_p and N_t must each be at least 2.")
    if strike <= 0:
        raise ValueError("Strike price must be positive.")
    if r < 0 or sig < 0:
        raise ValueError("Interest rate and volatility must be non-negative.")
    p = [float(x) for x in p]
    T = [float(x) for x in T]
    V = [[0.0 for _ in range(N_t)] for _ in range(N_p)]
    for i in range(N_p):
        V[i][-1] = max(p[i] - strike, 0.0)
    for j in range(N_t):
        V[0][j] = 0.0
    S_max = p[-1]
    e_const = 2.718281828459045
    for j in range(N_t):
        tau = T[-1] - T[j]
        V[-1][j] = S_max - strike * (e_const ** (-r * tau))
    return V
def construct_matrix(N_p, dp, dt, r, sig):
    """
    Constructs the tri-diagonal matrix used for an explicit (forward-Euler)
    finite-difference step of the Black–Scholes PDE.

    Parameters
    ----------
    N_p : int
        Total number of price-grid points, including the two boundaries.
    dp : float
        Spacing ΔS between adjacent price-grid points.
    dt : float
        Time-step size Δt.
    r  : float
        Risk-free interest rate.
    sig : float
        Volatility σ of the underlying asset.

    Returns
    -------
    scipy.sparse.csr_matrix
        (N_p − 2) × (N_p − 2) tri-diagonal matrix D such that
            V_int^{n+1} = D · V_int^{n},
        where V_int^{n} is the option-value vector at interior price nodes
        at time layer n.
    """
    if N_p < 3:
        raise ValueError("N_p must be at least 3 to provide interior nodes.")
    if dp <= 0 or dt <= 0:
        raise ValueError("dp and dt must be positive.")
    if r < 0 or sig < 0:
        raise ValueError("Interest rate r and volatility sig must be non-negative.")
    N_int = N_p - 2
    idx = np.arange(1, N_p - 1, dtype=float)
    S = idx * dp
    alpha = 0.5 * dt * (sig ** 2) * (S ** 2) / (dp ** 2)
    beta = 0.5 * dt * r * S / dp
    lower = alpha - beta
    main = 1 - 2 * alpha - r * dt
    upper = alpha + beta
    diagonals = [lower[1:], main, upper[:-1]]
    offsets = [-1, 0, 1]
    D = sparse.diags(diagonals, offsets, shape=(N_int, N_int), format='csr')
    return D
def forward_iteration(V, D, N_p, N_t, r, sig, dp, dt):
    '''
    Performs the backward-time finite-difference sweep to fill the option-price
    grid for all times.

    Parameters
    ----------
    V   : array_like, shape (N_p, N_t)
          Option-value grid.  The maturity column (j = N_t−1) and the two price
          boundary rows are already populated by `apply_boundary_conditions`.
    D   : scipy.sparse.csr_matrix, shape (N_p−2, N_p−2)
          Tri-diagonal matrix produced by `construct_matrix`, representing one
          backward time step for the interior price nodes.
    N_p : int   Number of price-grid points.
    N_t : int   Number of time-grid points.
    r   : float Risk-free interest rate.
    sig : float Volatility of the underlying asset.
    dp  : float Price-grid spacing ΔS.
    dt  : float Time-step size Δt.

    Returns
    -------
    numpy.ndarray
          The completed option-value grid with interior nodes filled for every
          time layer.
    '''
    V = np.asarray(V, dtype=float)
    N_int = N_p - 2
    if N_int < 1:
        return V
    idx = np.arange(1, N_p - 1, dtype=float)
    S = idx * dp
    alpha = 0.5 * dt * (sig ** 2) * (S ** 2) / (dp ** 2)
    beta = 0.5 * dt * r * S / dp
    upper_last_coeff = (alpha + beta)[-1]
    for j in range(N_t - 2, -1, -1):
        rhs = V[1:-1, j + 1].copy()
        rhs[-1] -= upper_last_coeff * V[-1, j]
        V[1:-1, j] = spsolve(D, rhs)
    return V
def price_option(price_step, time_step, strike, r, sig, max_price, min_price):
    '''
    Prices a European call option using the finite difference method.

    Parameters
    ----------
    price_step : int
        Number of grid points in the price (underlying-asset) dimension (N_p).
    time_step : int
        Number of grid points in the time dimension (N_t).
    strike : float
        Strike price of the European call option.
    r : float
        Risk-free interest rate (continuously compounded).
    sig : float
        Volatility of the underlying asset.
    max_price : float
        Upper bound for the price grid (numerical “infinity”).
    min_price : float
        Positive lower bound for the price grid (avoids S = 0).

    Returns
    -------
    numpy.ndarray
        Completed option-value grid V with shape (price_step, time_step).
        V[i, j] gives the option value at price grid point i and time grid j.
    '''
    p, dp, T, dt = initialize_grid(
        price_step, time_step,
        strike=strike,
        max_price=max_price,
        min_price=min_price
    )
    V = apply_boundary_conditions(
        price_step, time_step,
        p, T,
        strike, r, sig
    )
    D = construct_matrix(price_step, dp, dt, r, sig)
    V_filled = forward_iteration(
        V, D,
        price_step, time_step,
        r, sig, dp, dt
    )
    return np.asarray(V_filled, dtype=float)

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('63.5', 3)
target = targets[0]

price_step = 2000
time_step = 2000
strike = 500
r = 0.05
sig = 1
min_price = (1/5) * strike
max_price = 5 * strike
assert np.allclose(price_option(price_step, time_step, strike, r, sig, max_price, min_price), target)
target = targets[1]

price_step = 3000
time_step = 3000
strike = 600
r = 0.2
sig = 1
min_price = (1/5) * strike
max_price = 5 * strike
assert np.allclose(price_option(price_step, time_step, strike, r, sig, max_price, min_price), target)
target = targets[2]

price_step = 2500
time_step = 2500
strike = 5000
r = 0.5
sig = 5
min_price = (1/5) * strike
max_price = 5 * strike
assert np.allclose(price_option(price_step, time_step, strike, r, sig, max_price, min_price), target)
