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
