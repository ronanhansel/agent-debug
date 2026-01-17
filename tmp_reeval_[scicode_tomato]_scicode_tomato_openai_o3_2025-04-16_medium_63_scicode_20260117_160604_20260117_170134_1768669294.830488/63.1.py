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
