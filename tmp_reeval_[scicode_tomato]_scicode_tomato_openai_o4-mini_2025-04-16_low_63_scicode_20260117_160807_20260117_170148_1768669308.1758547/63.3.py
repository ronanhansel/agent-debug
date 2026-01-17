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

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('63.3', 3)
target = targets[0]

N_p=1000
r=0.2
sig=4
dt = 1
dp =1
assert np.allclose(construct_matrix(N_p,dp,dt,r,sig).toarray(), target)
target = targets[1]

N_p=3000
r=0.1
sig=10
dt = 1
dp =1
assert np.allclose(construct_matrix(N_p,dp,dt,r,sig).toarray(), target)
target = targets[2]

N_p=1000
r=0.5
sig=1
dt = 1
dp =1
assert np.allclose(construct_matrix(N_p,dp,dt,r,sig).toarray(), target)
