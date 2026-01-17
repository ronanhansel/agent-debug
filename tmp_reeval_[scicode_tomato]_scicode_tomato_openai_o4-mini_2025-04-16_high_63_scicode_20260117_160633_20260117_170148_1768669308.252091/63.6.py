import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def initialize_grid(price_step, time_step, strike, max_price, min_price):
    '''Initializes the grid for pricing a European call option.
    Inputs:
    price_step: Number of intervals in the price direction (int).
    time_step: Number of intervals in the time direction (int).
    strike: Strike price of the option (float).
    max_price: Upper bound for the asset price grid (float).
    min_price: Lower bound for the asset price grid (float).
    Outputs:
    p: 1D numpy array of price grid points from min_price to max_price, length price_step.
    dp: Uniform spacing between adjacent price points (float).
    T: 1D numpy array of time grid points from 0 to 1, length time_step.
    dt: Uniform spacing between adjacent time points (float).
    '''
    p = np.linspace(min_price, max_price, price_step)
    dp = p[1] - p[0]
    T = np.linspace(0.0, 1.0, time_step)
    dt = T[1] - T[0]
    return p, dp, T, dt
def apply_boundary_conditions(N_p, N_t, p, T, strike, r, sig):
    """
    Applies the boundary conditions on a price–time grid for a European call option.
    Inputs:
      N_p   : int, number of price grid points
      N_t   : int, number of time grid points
      p     : 1D numpy array of prices (length N_p)
      T     : 1D numpy array of times from 0 to T_m (length N_t)
      strike: float, option strike price
      r     : float, risk-free interest rate
      sig   : float, volatility (unused in boundary setup)
    Output:
      V     : 2D numpy array of shape (N_p, N_t) with boundary conditions applied
    """
    V = np.zeros((N_p, N_t))
    T_m = T[-1]
    V[0, :] = 0.0
    p_max = p[-1]
    tau = T_m - T
    V[-1, :] = p_max - strike * np.exp(-r * tau)
    V[:, -1] = np.maximum(p - strike, 0.0)
    return V
def construct_matrix(N_p, dp, dt, r, sig):
    '''Constructs the tri-diagonal matrix for the finite difference method.
    Inputs:
    N_p: The number of grid points in the price direction. (int)
    dp: The spacing between adjacent price grid points. (float)
    dt: The spacing between adjacent time grid points. (float)
    r: The risk-free interest rate. (float)
    sig: The volatility of the underlying asset. (float)
    Outputs:
    D: The tri-diagonal time-stepping matrix. Shape: (N_p-2, N_p-2)
    '''
    # number of interior nodes
    M = N_p - 2

    # interior price grid S_i = i * dp, i = 1..N_p-2
    S = np.arange(1, N_p-1) * dp

    # finite-difference coefficients at each interior node
    alpha = 0.5 * sig**2 * S**2 / dp**2 - 0.5 * r * S / dp
    beta  = - sig**2 * S**2 / dp**2 - r
    gamma = 0.5 * sig**2 * S**2 / dp**2 + 0.5 * r * S / dp

    # build the three diagonals for the tri-diagonal operator
    sub_diag   = alpha[1:]      # length M-1, goes on offset -1
    main_diag  = beta           # length M,   goes on offset  0
    super_diag = gamma[:-1]     # length M-1, goes on offset +1

    # sparse tri-diagonal matrix A
    A = sparse.diags(
        diagonals=[sub_diag, main_diag, super_diag],
        offsets=[-1, 0, 1],
        shape=(M, M),
        format='csc'
    )

    # time-stepping matrix D = I + dt * A
    I = sparse.eye(M, format='csc')
    D = I + dt * A

    return D
def forward_iteration(V, D, N_p, N_t, r, sig, dp, dt):
    for j in range(N_t - 2, -1, -1):
        b = V[1:-1, j + 1]
        x = spsolve(D, b)
        V[1:-1, j] = x
    return V
def price_option(price_step, time_step, strike, r, sig, max_price, min_price):
    '''Prices a European call option using the finite difference method.
    Inputs:
      price_step: int, number of price grid points (N_p)
      time_step:  int, number of time grid points (N_t)
      strike:     float, strike price of the option
      r:          float, risk-free interest rate
      sig:        float, volatility of the underlying asset
      max_price:  float, upper bound for the price grid
      min_price:  float, lower bound for the price grid
    Outputs:
      V: 2D numpy array of shape (N_p, N_t) containing the option values
    '''
    N_p, N_t = price_step, time_step
    p, dp, T, dt = initialize_grid(N_p, N_t, strike, max_price, min_price)
    V = apply_boundary_conditions(N_p, N_t, p, T, strike, r, sig)
    D = construct_matrix(N_p, dp, dt, r, sig)
    V = forward_iteration(V, D, N_p, N_t, r, sig, dp, dt)
    return V
def price_option_of_time(price_step, time_step, strike, r, sig, max_price, min_price, t, S0):
    p, dp, T, dt = initialize_grid(price_step, time_step, strike, max_price, min_price)
    V = price_option(price_step, time_step, strike, r, sig, max_price, min_price)
    jf = t * (time_step - 1)
    jl = int(np.floor(jf))
    ju = min(jl + 1, time_step - 1)
    wt = (jf - jl) if ju > jl else 0.0
    if max_price > min_price:
        if S0 <= min_price:
            il = iu = 0
            wp = 0.0
        elif S0 >= max_price:
            il = iu = price_step - 1
            wp = 0.0
        else:
            pf = (S0 - min_price) * (price_step - 1) / (max_price - min_price)
            il = int(np.floor(pf))
            iu = min(il + 1, price_step - 1)
            wp = (pf - il) if iu > il else 0.0
    else:
        il = iu = 0
        wp = 0.0
    V_ll = V[il, jl]
    V_ul = V[iu, jl]
    V_lu = V[il, ju]
    V_uu = V[iu, ju]
    V_low = V_ll + wp * (V_ul - V_ll)
    V_high = V_lu + wp * (V_uu - V_lu)
    return V_low + wt * (V_high - V_low)

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('63.6', 3)
target = targets[0]

price_step = 3000  # Number of price steps
time_step = 3000   # Number of time steps
strike = 1000      # Strike price of the option
r = 0.05          # Risk-free interest rate
sig = 1         # Volatility of the underlying asset
S0 = 100          # Initial stock price
max_price = 5 * strike # maximum bound on price grid
min_price = (1/5) * strike # minimum bound on price grid
t = 0
assert np.allclose(price_option_of_time(price_step, time_step, strike, r, sig, max_price, min_price, t, S0), target)
target = targets[1]

price_step = 3000  # Number of price steps
time_step = 3000   # Number of time steps
strike = 1000      # Strike price of the option
r = 0.05          # Risk-free interest rate
sig = 1         # Volatility of the underlying asset
S0 = 500          # Initial stock price
max_price = 5 * strike # maximum bound on price grid
min_price = (1/5) * strike # minimum bound on price grid
t = 0.5
assert np.allclose(price_option_of_time(price_step, time_step, strike, r, sig, max_price, min_price, t, S0), target)
target = targets[2]

price_step = 3000  # Number of price steps
time_step = 3000   # Number of time steps
strike = 3000      # Strike price of the option
r = 0.2          # Risk-free interest rate
sig = 1         # Volatility of the underlying asset
S0 = 1000        # Initial stock price
max_price = 5 * strike # maximum bound on price grid
min_price = (1/5) * strike # minimum bound on price grid
t = 0.5
assert np.allclose(price_option_of_time(price_step, time_step, strike, r, sig, max_price, min_price, t, S0), target)
