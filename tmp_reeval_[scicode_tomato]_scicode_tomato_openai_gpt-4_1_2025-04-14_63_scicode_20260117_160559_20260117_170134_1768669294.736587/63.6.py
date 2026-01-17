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
def price_option_of_time(price_step, time_step, strike, r, sig, max_price, min_price, t, S0):
    N_p = price_step
    N_t = time_step
    K = strike
    S_grid = np.linspace(min_price, max_price, N_p)
    dS = S_grid[1] - S_grid[0]
    T = 1.0
    t_grid = np.linspace(0, T, N_t)
    dt = t_grid[1] - t_grid[0]
    V = np.zeros((N_p, N_t))
    V[:, -1] = np.maximum(S_grid - K, 0)
    for j in range(N_t):
        tau = t_grid[j]
        V[0, j] = 0
        V[-1, j] = max_price - K * np.exp(-r * (T - tau))
    size = N_p - 2
    S_frac = S_grid[1:-1] / dS
    a = 0.5 * dt * ((sig ** 2) * (S_frac) ** 2 - r * S_frac)
    b = 1 + dt * ((sig ** 2) * (S_frac) ** 2 + r)
    c = -0.5 * dt * ((sig ** 2) * (S_frac) ** 2 + r * S_frac)
    for n in range(N_t - 2, -1, -1):
        lower = -a[1:]
        main = b
        upper = -c[:-1]
        diagonals = [lower, main, upper]
        D = sparse.diags(diagonals, offsets=[-1, 0, 1], shape=(size, size), format='csr')
        rhs = V[1:-1, n + 1].copy()
        rhs[0]   -= a[0]   * V[0,   n]
        rhs[-1]  -= c[-1]  * V[-1,  n]
        V[1:-1, n] = spsolve(D, rhs)
    t_pos = t * (N_t - 1)
    t_low_idx = int(np.floor(t_pos))
    t_high_idx = min(t_low_idx + 1, N_t - 1)
    t_w = t_pos - t_low_idx
    S0 = np.clip(S0, min_price, max_price)
    s_pos = (S0 - min_price) / (max_price - min_price) * (N_p - 1)
    s_low_idx = int(np.floor(s_pos))
    s_high_idx = min(s_low_idx + 1, N_p - 1)
    s_w = s_pos - s_low_idx
    f00 = V[s_low_idx, t_low_idx]
    f10 = V[s_high_idx, t_low_idx]
    f01 = V[s_low_idx, t_high_idx]
    f11 = V[s_high_idx, t_high_idx]
    f0 = f00 * (1 - s_w) + f10 * s_w
    f1 = f01 * (1 - s_w) + f11 * s_w
    Price = f0 * (1 - t_w) + f1 * t_w
    return float(Price)

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
