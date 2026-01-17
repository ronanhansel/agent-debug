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

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('63.2', 3)
target = targets[0]

N_p=1000
N_t=2000
r=0.02
sig=2
dt = 1
dp =1
strike = 1000
min_price = 300
max_price = 2500
p, dp, T, dt = initialize_grid(N_p,N_t,strike, min_price, max_price)
assert np.allclose(apply_boundary_conditions(N_p,N_t,p,T,strike,r,sig), target)
target = targets[1]

N_p=4000
N_t=4000
r=0.2
sig=1
dt = 1
dp =1
strike = 1000
min_price = 100
max_price = 2500
p, dp, T, dt = initialize_grid(N_p,N_t,strike, min_price, max_price)
assert np.allclose(apply_boundary_conditions(N_p,N_t,p,T,strike,r,sig), target)
target = targets[2]

N_p=1000
N_t=2000
r=0.5
sig=1
dt = 1
dp =1
strike = 1000
min_price = 100
max_price = 2500
p, dp, T, dt = initialize_grid(N_p,N_t,strike, min_price, max_price)
assert np.allclose(apply_boundary_conditions(N_p,N_t,p,T,strike,r,sig), target)
