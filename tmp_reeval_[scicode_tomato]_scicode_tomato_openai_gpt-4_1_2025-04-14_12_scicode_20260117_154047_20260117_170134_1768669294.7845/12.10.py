# Compatibility shim for scipy.integrate.simps (deprecated in SciPy 1.14, removed in 1.17)
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

from scipy import integrate
from scipy import optimize
import numpy as np

def f_Schrod(energy, l, r_grid):
    safe_r = r_grid.copy()
    safe_r[safe_r == 0] = 1e-12
    f_r = -2.0 / safe_r + l * (l + 1) / (safe_r ** 2) - 2.0 * energy
    return f_r
def Numerov(f_in, u_at_0, up_at_0, step):
    N = len(f_in)
    h = step
    u = [0.0]*N
    u[0] = u_at_0
    if N > 1:
        u[1] = u_at_0 + h * up_at_0 + 0.5 * h * h * f_in[0] * u_at_0
    for n in range(1, N - 1):
        h2 = h * h
        denom = 1 + (h2 / 12.0) * f_in[n + 1]
        num = (2 * u[n] * (1 - (5 * h2 / 12.0) * f_in[n])
               - u[n - 1] * (1 + (h2 / 12.0) * f_in[n - 1]))
        u[n + 1] = num / denom
    return u
def compute_Schrod(energy, r_grid, l):
    def f_Schrod_py(energy, l, r_grid):
        f_r = []
        for r in r_grid:
            rp = r if abs(r) > 1e-12 else 1e-12
            val = -2.0 / rp + l * (l+1) / (rp**2) - 2.0 * energy
            f_r.append(val)
        return f_r

    def Numerov_py(f_in, u_at_0, up_at_0, step):
        N = len(f_in)
        h = step
        u = [0.0] * N
        u[0] = u_at_0
        if N > 1:
            u[1] = u_at_0 + h * up_at_0 + 0.5 * h * h * f_in[0] * u_at_0
        for n in range(1, N - 1):
            h2 = h * h
            denom = 1 + (h2 / 12.0) * f_in[n + 1]
            num = (2 * u[n] * (1 - (5 * h2 / 12.0) * f_in[n])
                   - u[n - 1] * (1 + (h2 / 12.0) * f_in[n - 1]))
            u[n + 1] = num / denom
        return u

    def simpsons_rule(x, y):
        n = len(x)
        if n < 3 or n % 2 == 0:
            acc = 0.0
            for i in range(n-1):
                acc += 0.5 * (x[i+1]-x[i]) * (y[i+1] + y[i])
            return acc
        else:
            h = (x[-1] - x[0]) / (n - 1)
            s = y[0] + y[-1]
            for i in range(1, n-1):
                if i % 2 == 0:
                    s += 2 * y[i]
                else:
                    s += 4 * y[i]
            return s * h / 3.0

    r_grid_rev = list(reversed(r_grid))
    f_r_rev = f_Schrod_py(energy, l, r_grid_rev)
    if len(r_grid) > 1:
        step = r_grid_rev[0] - r_grid_rev[1]
    else:
        step = 1.0

    u_at_0 = 0.0
    up_at_0 = -1e-7

    u_rev = Numerov_py(f_r_rev, u_at_0, up_at_0, step)
    u = list(reversed(u_rev))

    prob_density = [abs(ui)**2 for ui in u]
    norm = simpsons_rule(r_grid, prob_density)
    if norm != 0:
        ur_norm = [ui / norm**0.5 for ui in u]
    else:
        ur_norm = u
    return ur_norm
def shoot(energy, r_grid, l):
    u = compute_Schrod(energy, r_grid, l)
    r0, r1 = r_grid[0], r_grid[1]
    u0, u1 = u[0], u[1]
    r0_safe = r0 if abs(r0) > 1e-14 else 1e-14
    r1_safe = r1 if abs(r1) > 1e-14 else 1e-14
    w0 = u0 / (r0_safe ** l) if l > 0 else u0
    w1 = u1 / (r1_safe ** l) if l > 0 else u1
    if abs(r1 - r0) < 1e-14:
        f_at_0 = w0
    else:
        f_at_0 = w0 + (w1 - w0) * (-r0) / (r1 - r0)
    return f_at_0
def find_bound_states(r_grid, l, energy_grid):
    bound_states = []
    max_states = 10
    shoot_vals = [shoot(e, r_grid, l) for e in energy_grid]
    for i in range(len(energy_grid) - 1):
        S1 = shoot_vals[i]
        S2 = shoot_vals[i + 1]
        E1 = energy_grid[i]
        E2 = energy_grid[i + 1]
        if S1 * S2 < 0:
            def shoot_func(E):
                return shoot(E, r_grid, l)
            try:
                root = optimize.brentq(shoot_func, E1, E2, xtol=1e-9, rtol=1e-7, maxiter=100)
                bound_states.append((l, root))
            except Exception:
                continue
            if len(bound_states) >= max_states:
                break
        elif S1 == 0.0:
            bound_states.append((l, E1))
            if len(bound_states) >= max_states:
                break
    return bound_states
def sort_states(bound_states):
    '''
    Input
    bound_states: a list of bound states found by the find_bound_states function, each element is a tuple containing the angular momentum quantum number (int) and energy (float)
    Output
    sorted_states: a list that contains the sorted bound_states tuples according to the following rules: State with lower energy will be in front. If two states have the same energy, the one with smaller angular momentum quantum number will be in front.
    '''
    sorted_states = sorted(bound_states, key=lambda x: (x[1], x[0]))
    return sorted_states
def calculate_charge_density(bound_states, r_grid, Z):
    sorted_states = sort_states(bound_states)
    total_charge_density = [0.0 for _ in r_grid]
    electrons_remaining = int(Z)
    for (l, energy) in sorted_states:
        g_l = 2 * (2 * l + 1)
        n_fill = min(g_l, electrons_remaining)
        if n_fill == 0:
            break
        u_r = compute_Schrod(energy, r_grid, l)
        u_r = [abs(val) for val in u_r]
        density = [n_fill * (u**2) for u in u_r]
        total_charge_density = [a + b for a, b in zip(total_charge_density, density)]
        electrons_remaining -= n_fill
        if electrons_remaining == 0:
            break
    charge_density = total_charge_density
    return charge_density
def calculate_HartreeU(charge_density, u_at_0, up_at_0, step, r_grid, Z):
    safe_r = list(r_grid)
    for i in range(len(safe_r)):
        if safe_r[i] == 0:
            safe_r[i] = 1e-12

    f_in = [-8.0 * 3.141592653589793 * safe_r[i] * charge_density[i] for i in range(len(safe_r))]
    N = len(safe_r)
    h = step
    U = [0.0] * N
    U[0] = u_at_0
    if N > 1:
        U[1] = u_at_0 + h * up_at_0 + 0.5 * h * h * f_in[0]
    for n in range(1, N - 1):
        h2 = h * h
        U[n + 1] = (2 * U[n] - U[n - 1] +
                    (h2 / 12.0) * (f_in[n + 1] + 10 * f_in[n] + f_in[n - 1]))
    return U
def f_Schrod_Hartree(energy, r_grid, l, Z, hartreeU):
    f_r = []
    for i in range(len(r_grid)):
        r = r_grid[i]
        r_safe = r if abs(r) > 1e-12 else 1e-12
        U = hartreeU[i]
        V_H = U / r_safe
        val = -2.0 * Z / r_safe + l * (l + 1) / (r_safe ** 2) + 2.0 * V_H - 2.0 * energy
        f_r.append(val)
    return f_r
def compute_Schrod_Hartree(energy, r_grid, l, Z, hartreeU):
    r_grid_rev = np.array(list(reversed(r_grid)))
    hartreeU_rev = np.array(list(reversed(hartreeU)))
    f_r_rev = f_Schrod_Hartree(energy, r_grid_rev, l, Z, hartreeU_rev)
    if len(r_grid_rev) > 1:
        step = r_grid_rev[0] - r_grid_rev[1]
    else:
        step = 1.0
    u_at_0 = 0.0
    up_at_0 = -1e-7
    N = len(r_grid_rev)
    u_rev = [0.0] * N
    u_rev[0] = u_at_0
    if N > 1:
        u_rev[1] = u_at_0 + step * up_at_0 + 0.5 * step * step * f_r_rev[0] * u_at_0
    for n in range(1, N - 1):
        h2 = step * step
        denom = 1 + (h2 / 12.0) * f_r_rev[n + 1]
        num = (2 * u_rev[n] * (1 - (5 * h2 / 12.0) * f_r_rev[n])
               - u_rev[n - 1] * (1 + (h2 / 12.0) * f_r_rev[n - 1]))
        u_rev[n + 1] = num / denom
    u = np.array(list(reversed(u_rev)))
    prob_density = np.abs(u) ** 2
    try:
        norm = integrate.simpson(prob_density, r_grid)
    except Exception:
        norm = integrate.simps(prob_density, r_grid)
    if norm != 0:
        ur_norm = u / np.sqrt(norm)
    else:
        ur_norm = u
    return ur_norm

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('12.10', 3)
target = targets[0]

energy_grid = -1.2/np.arange(1,20,0.2)**2
r_grid = np.linspace(1e-8,100,2000)
Z=28
nmax = 5
bound_states=[]
for l in range(nmax):
    bound_states += find_bound_states(r_grid, l, energy_grid)
charge_density = calculate_charge_density(bound_states,r_grid,Z)
hu = calculate_HartreeU(charge_density, 0.0, 0.5, r_grid[0]-r_grid[1], r_grid, Z)
assert np.allclose(compute_Schrod_Hartree(-0.5, r_grid, 2, Z, hu), target)
target = targets[1]

energy_grid = -1.2/np.arange(1,20,0.2)**2
r_grid = np.linspace(1e-8,100,2000)
Z=14
nmax = 3
bound_states=[]
for l in range(nmax):
    bound_states += find_bound_states(r_grid, l, energy_grid)
charge_density = calculate_charge_density(bound_states,r_grid,Z)
hu = calculate_HartreeU(charge_density, 0.0, 0.5, r_grid[0]-r_grid[1], r_grid, Z)
assert np.allclose(compute_Schrod_Hartree(-0.4, r_grid, 2, Z, hu), target)
target = targets[2]

energy_grid = -0.9/np.arange(1,20,0.2)**2
r_grid = np.linspace(1e-8,100,2000)
Z=14
nmax = 5
bound_states=[]
for l in range(nmax):
    bound_states += find_bound_states(r_grid, l, energy_grid)
charge_density = calculate_charge_density(bound_states,r_grid,Z)
hu = calculate_HartreeU(charge_density, 0.0, 0.5, r_grid[0]-r_grid[1], r_grid, Z)
assert np.allclose(compute_Schrod_Hartree(-0.5, r_grid, 3, Z, hu), target)
