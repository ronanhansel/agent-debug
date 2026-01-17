import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def initialize_grid(price_step, time_step, strike, max_price, min_price):
    '''Initializes the grid for pricing a European call option.
    Inputs:
        price_step: The number of steps or intervals in the price direction. (int)
        time_step: The number of steps or intervals in the time direction. (int)
        strike: The strike price of the European call option. (float)
        max_price: upper bound for the price grid. (float)
        min_price: lower bound for the price grid. (float)
    Outputs:
        p: An array containing the grid points for prices (shape: price_step).
        dp: The spacing between adjacent price grid points. (float)
        T: An array containing the grid points for time from 0 to 1 (shape: time_step).
        dt: The spacing between adjacent time grid points. (float)
    '''
    p = np.linspace(min_price, max_price, price_step)
    dp = (p[1] - p[0]) if price_step > 1 else 0.0
    T = np.linspace(0.0, 1.0, time_step)
    dt = (T[1] - T[0]) if time_step > 1 else 0.0
    return p, dp, T, dt
def apply_boundary_conditions(N_p, N_t, p, T, strike, r, sig):
    """
    Applies the boundary conditions for a European call option on a finite-difference grid.

    Inputs:
    N_p   : number of grid points in the price direction (int)
    N_t   : number of grid points in the time direction (int)
    p     : NumPy array of price grid points, shape (N_p,)
    T     : NumPy array of time grid points, shape (N_t,)
    strike: strike price of the call option (float)
    r     : risk-free interest rate (float)
    sig   : volatility of the underlying stock (float)

    Output:
    V     : 2D NumPy array of option values with boundary conditions applied, shape (N_p, N_t)
    """
    V = np.zeros((N_p, N_t))
    T_final = T[-1]
    tau = T_final - T
    V[:, -1] = np.maximum(p - strike, 0.0)
    V[0, :] = 0.0
    V[-1, :] = p[-1] - strike * np.exp(-r * tau)
    return V
def construct_matrix(N_p, dp, dt, r, sig):
    """Constructs the tri-diagonal matrix for the finite difference method.
    Inputs:
      N_p: number of price grid points (int)
      dp:  price step size (float)
      dt:  time step size (float)
      r:   risk-free interest rate (float)
      sig: volatility of the underlying asset (float)
    Output:
      D: sparse tridiagonal matrix of shape (N_p-2, N_p-2)
    """
    N = N_p - 2
    if N <= 0:
        return sparse.csc_matrix((0, 0))
    j = np.arange(1, N_p - 1)
    p_j = j * dp
    A = 0.5 * sig**2 * (p_j**2) / (dp**2)
    B = (r * p_j) / (2.0 * dp)
    a = dt * (A - B)
    b = 1.0 - dt * (2.0 * A + r)
    c = dt * (A + B)
    diagonals = [a[1:], b, c[:-1]]
    offsets = [-1, 0, 1]
    D = sparse.diags(diagonals, offsets, shape=(N, N), format='csc')
    return D
def forward_iteration(V, D, N_p, N_t, r, sig, dp, dt):
    M = N_p - 2
    if M <= 0 or N_t < 2:
        return V
    for t in range(N_t - 2, -1, -1):
        rhs = V[1:N_p-1, t+1]
        x = spsolve(D, rhs)
        V[1:N_p-1, t] = x
    return V
def price_option(price_step, time_step, strike, r, sig, max_price, min_price):
    """
    Prices a European call option using the finite difference method.
    Inputs:
        price_step : number of steps in price direction (N_p)
        time_step  : number of steps in time direction (N_t)
        strike     : strike price of the option
        r          : risk-free interest rate
        sig        : volatility of the underlying asset
        max_price  : upper bound for the price grid
        min_price  : lower bound for the price grid
    Output:
        V : 2D NumPy array of option values, shape (N_p, N_t)
    """
    N_p = price_step
    N_t = time_step
    p, dp, T, dt = initialize_grid(N_p, N_t, strike, max_price, min_price)
    V = apply_boundary_conditions(N_p, N_t, p, T, strike, r, sig)
    D = construct_matrix(N_p, dp, dt, r, sig)
    V = forward_iteration(V, D, N_p, N_t, r, sig, dp, dt)
    return V
def price_option_of_time(price_step, time_step, strike, r, sig, max_price, min_price, t, S0):
    t = max(0.0, min(t, 1.0))
    S0 = max(min_price, min(S0, max_price))
    p, dp, T, dt = initialize_grid(price_step, time_step, strike, max_price, min_price)
    V = price_option(price_step, time_step, strike, r, sig, max_price, min_price)
    max_t_idx = time_step - 1
    t_float = t * max_t_idx
    i_t0 = int(t_float)
    i_t1 = min(i_t0 + 1, max_t_idx)
    w_t = t_float - i_t0
    p_float = (S0 - min_price) / dp
    p_float = max(0.0, min(p_float, price_step - 1.0))
    i_p0 = int(p_float)
    i_p1 = min(i_p0 + 1, price_step - 1)
    w_p = p_float - i_p0
    v00 = (1 - w_p) * V[i_p0, i_t0] + w_p * V[i_p1, i_t0]
    v01 = (1 - w_p) * V[i_p0, i_t1] + w_p * V[i_p1, i_t1]
    price = (1 - w_t) * v00 + w_t * v01
    return price

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
