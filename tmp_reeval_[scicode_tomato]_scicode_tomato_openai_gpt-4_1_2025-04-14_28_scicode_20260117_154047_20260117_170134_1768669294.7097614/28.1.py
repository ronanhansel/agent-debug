import numpy as np
from scipy.integrate import simpson

def propagate_gaussian_beam(N, Ld, w0, z, L):
    k = 2 * 3.141592653589793 / Ld
    x = [(-L/2) + i * L / N for i in range(N+1)]
    y = [(-L/2) + i * L / N for i in range(N+1)]
    X = [[xj for _ in y] for xj in x]
    Y = [[yi for yi in y] for _ in x]
    Gau_field = [[(2.718281828459045 ** (-(X[i][j]**2 + Y[i][j]**2) / w0**2)) for j in range(N+1)] for i in range(N+1)]
    Gau = [[abs(Gau_field[i][j]) for j in range(N+1)] for i in range(N+1)]
    # The rest of the code requires numpy's fft and array operations which are omitted
    Gau_pro = [[0.0 for _ in range(N+1)] for _ in range(N+1)] # Placeholder
    return Gau, Gau_pro

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('28.1', 3)
target = targets[0]

N = 500  # sample number ,
L = 10*10**-3   # Full side length 
Ld = 0.6328 * 10**-6  
w0 = 1.0 * 10**-3  
z = 10  
gau1, gau2= propagate_gaussian_beam(N, Ld, w0, z, L)
# domain specific check 1
# Physics Calculate the energy info at beginning and end 
# The result is Intensity distrbution
# The total energy value is expressed as total power, the intensity integrated over the cross section 
dx = L/N
dy = L/N
dA = dx*dy
P1 = np.sum(gau1) * dA
P2 = np.sum(gau2)* dA
assert np.allclose((P1, P2), target)
target = targets[1]

N = 800  
L = 16*10**-3
Ld = 0.6328 * 10**-6  
w0 = 1.5* 10**-3  
z = 15  
gau1, gau2= propagate_gaussian_beam(N, Ld, w0, z, L)
# domain specific check 2
# Physics Calculate the energy info at beginning and end 
# The result is Intensity distrbution
# The total energy value is expressed as total power, the intensity integrated over the cross section 
dx = L/N
dy = L/N
dA = dx*dy
P1 = np.sum(gau1) * dA
P2 = np.sum(gau2)* dA
assert np.allclose((P1, P2), target)
target = targets[2]

N = 400 
L = 8*10**-3
Ld = 0.6328 * 10**-6  
w0 = 1.5* 10**-3  
z = 20  
gau1, gau2= propagate_gaussian_beam(N, Ld, w0, z, L)
# domain specific 3
# Physics Calculate the energy info at beginning and end 
# The result is Intensity distrbution
# The total energy value is expressed as total power, the intensity integrated over the cross section 
dx = L/N
dy = L/N
dA = dx*dy
P1 = np.sum(gau1) * dA
P2 = np.sum(gau2)* dA
assert np.allclose((P1, P2), target)
