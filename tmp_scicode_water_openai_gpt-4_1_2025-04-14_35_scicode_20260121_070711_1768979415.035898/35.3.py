import numpy as np
import itertools

def ground_state_wavelength(L, mr):
    hbar = 1.054571817e-34
    c = 299792458
    pi = 3.141592653589793
    E1 = (pi**2 * hbar**2) / (2 * mr * L**2)
    wavelength = (hbar * c * 2 * pi) / E1
    return wavelength
def calculate_force(mass, acceleration):
    return mass * acceleration
def largest_photon_wavelengths(a, b, c, mr, N):
    h = 6.62607015e-34
    c0 = 2.99792458e8
    e = 1.602176634e-19
    m0 = 9.10938356e-31
    a = a * 1e-9
    b = b * 1e-9
    c = c * 1e-9
    levels = []
    for nx in range(1, N+2):
        for ny in range(1, N+2):
            for nz in range(1, N+2):
                En = (h**2/8) * (nx**2/(mr*m0*a**2) + ny**2/(mr*m0*b**2) + nz**2/(mr*m0*c**2))
                levels.append(En)
    levels = sorted(set(levels))
    E0 = levels[0]
    wavelengths = []
    for i in range(1, N+1):
        deltaE = levels[i] - E0
        wl = h*c0/(deltaE)
        wavelengths.append(wl*1e9)
    wavelengths = sorted(wavelengths, reverse=True)
    return wavelengths

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('35.3', 4)
target = targets[0]

A = absorption(0.6,3,4,10**6,5)
assert (all(i>10**10 for i in A)) == target
target = targets[1]

A = absorption(0.3,7,3,5,10)
assert np.allclose(sorted(A)[::-1], target)
target = targets[2]

A = absorption(0.6,3,4,5,5)
assert np.allclose(sorted(A)[::-1], target)
target = targets[3]

A = absorption(0.6,37,23,18,10)
assert np.allclose(sorted(A)[::-1], target)
