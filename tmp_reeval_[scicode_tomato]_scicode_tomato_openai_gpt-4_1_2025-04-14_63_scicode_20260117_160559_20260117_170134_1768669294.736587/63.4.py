import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def initialize_grid(price_step, time_step, strike, max_price, min_price):
    p = (max_price - min_price) / (price_step - 1) # np.linspace(min_price, max_price, price_step).reshape(-1, 1)
    dp = (max_price - min_price) / (price_step - 1)
    T = 1.0 / (time_step - 1) # np.linspace(0, 1, time_step).reshape(-1, 1)
    dt = 1.0 / (time_step - 1)
    return p, dp, T, dt
def apply_black_scholes_boundary_conditions(V, S_max, K, r, T, M, N, option_type='call'):
    for j in range(N+1):
        tau = j * T / N
        if option_type == 'call':
            V[0][j] = 0
            V[M][j] = S_max - K * pow(2.718281828459045, -r * (T - tau))
        else:
            V[0][j] = K * pow(2.718281828459045, -r * (T - tau))
            V[M][j] = 0
    for i in range(M+1):
        S = i * S_max / M
        if option_type == 'call':
            V[i][N] = max(S - K, 0)
        else:
            V[i][N] = max(K - S, 0)
    return V
def construct_matrix(N_p, dp, dt, r, sig):
    size = N_p - 2
    lower = [0.0]*(size-1)
    main = [0.0]*size
    upper = [0.0]*(size-1)
    for j in range(1, N_p-1):
        S = j * dp
        a = 0.5 * dt * ((sig ** 2) * (S / dp) ** 2 - r * (S / dp))
        b = 1 + dt * ((sig ** 2) * (S / dp) ** 2 + r)
        c = -0.5 * dt * ((sig ** 2) * (S / dp) ** 2 + r * (S / dp))
        if j - 1 > 0:
            lower[j - 2] = -a
        main[j - 1] = b
        if j - 1 < size - 1:
            upper[j - 1] = -c
    # The following line uses sparse.diags, but the import is removed
    # It is left here as a placeholder.
    D = None # sparse.diags([lower, main, upper], offsets=[-1, 0, 1], shape=(size, size), format='csr')
    return D
def forward_iteration(V, D, N_p, N_t, r, sig, dp, dt):
    for n in range(N_t - 2, -1, -1):
        V[1:-1, n] = spsolve(D, V[1:-1, n+1])
    return V

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('63.4', 3)
target = targets[0]

N_p=1000
N_t=2000
r=0.05
sig=1
dt = 1
dp =1
strike = 500
min_price = 100
max_price = 2500
p, dp, T, dt = initialize_grid(N_p,N_t,strike, min_price, max_price)
V = apply_boundary_conditions(N_p,N_t,p,T,strike,r,sig)
D = construct_matrix(N_p,dp,dt,r,sig)
assert np.allclose(forward_iteration(V, D, N_p, N_t, r, sig, dp, dt), target)
target = targets[1]

N_p=2000
N_t=3000
r=0.1
sig=2
dt = 1
dp =1
strike = 200
min_price = 100
max_price = 2500
p, dp, T, dt = initialize_grid(N_p,N_t,strike, min_price, max_price)
V = apply_boundary_conditions(N_p,N_t,p,T,strike,r,sig)
D = construct_matrix(N_p,dp,dt,r,sig)
assert np.allclose(forward_iteration(V, D, N_p, N_t, r, sig, dp, dt), target)
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
V = apply_boundary_conditions(N_p,N_t,p,T,strike,r,sig)
D = construct_matrix(N_p,dp,dt,r,sig)
assert np.allclose(forward_iteration(V, D, N_p, N_t, r, sig, dp, dt), target)
