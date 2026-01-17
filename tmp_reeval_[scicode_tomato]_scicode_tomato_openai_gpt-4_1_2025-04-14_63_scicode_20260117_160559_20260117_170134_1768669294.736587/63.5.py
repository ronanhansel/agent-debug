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
def price_option(price_step, time_step, strike, r, sig, max_price, min_price):
    N_p = price_step
    N_t = time_step
    K = strike
    S_grid = [min_price + i * (max_price - min_price) / (N_p - 1) for i in range(N_p)]
    dp = S_grid[1] - S_grid[0]
    T = 1.0
    t_grid = [i * T / (N_t - 1) for i in range(N_t)]
    dt = t_grid[1] - t_grid[0]
    V = [[0.0 for _ in range(N_t)] for _ in range(N_p)]
    for j in range(N_t):
        tau = t_grid[j]
        V[0][j] = 0
        V[-1][j] = max_price - K
    for i in range(N_p):
        S = S_grid[i]
        V[i][-1] = max(S - K, 0)
    size = N_p - 2
    lower = [0.0] * (size - 1)
    main = [0.0] * size
    upper = [0.0] * (size - 1)
    for j in range(1, N_p - 1):
        S = S_grid[j]
        a = 0.5 * dt * ((sig ** 2) * (S / dp) ** 2 - r * (S / dp))
        b = 1 + dt * ((sig ** 2) * (S / dp) ** 2 + r)
        c = -0.5 * dt * ((sig ** 2) * (S / dp) ** 2 + r * (S / dp))
        if j - 1 > 0:
            lower[j - 2] = -a
        main[j - 1] = b
        if j - 1 < size - 1:
            upper[j - 1] = -c
    for n in range(N_t - 2, -1, -1):
        rhs = [V[j][n + 1] for j in range(1, N_p - 1)]
        if size > 0:
            rhs[0] += lower[0] * V[0][n]
            rhs[-1] += upper[-1] * V[-1][n]
        A = [[0.0 for _ in range(size)] for _ in range(size)]
        for i in range(size):
            A[i][i] = main[i]
            if i > 0:
                A[i][i - 1] = lower[i - 1]
            if i < size - 1:
                A[i][i + 1] = upper[i]
        x = [0.0 for _ in range(size)]
        for i in range(size):
            for j in range(size):
                if A[i][j] != 0 and i != j:
                    rhs[i] -= A[i][j] * x[j]
            x[i] = rhs[i] / A[i][i] if A[i][i] != 0 else 0
        for idx in range(size):
            V[idx + 1][n] = x[idx]
    return V

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
