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
