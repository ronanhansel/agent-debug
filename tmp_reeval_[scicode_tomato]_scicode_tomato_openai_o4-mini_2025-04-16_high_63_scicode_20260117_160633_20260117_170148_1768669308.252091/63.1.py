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

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('63.1', 3)
target = targets[0]

from scicode.compare.cmp import cmp_tuple_or_list
price_step = 5000
time_step = 2000
strike = 100
min_price = 20
max_price = 500
assert cmp_tuple_or_list(initialize_grid(price_step,time_step,strike, min_price, max_price), target)
target = targets[1]

from scicode.compare.cmp import cmp_tuple_or_list
price_step = 3000
time_step = 2000
strike = 500
min_price = 100
max_price = 2500
assert cmp_tuple_or_list(initialize_grid(price_step,time_step,strike, min_price, max_price), target)
target = targets[2]

from scicode.compare.cmp import cmp_tuple_or_list
price_step = 3000
time_step = 2000
strike = 50
min_price = 10
max_price = 250
assert cmp_tuple_or_list(initialize_grid(price_step,time_step,strike, min_price, max_price), target)
