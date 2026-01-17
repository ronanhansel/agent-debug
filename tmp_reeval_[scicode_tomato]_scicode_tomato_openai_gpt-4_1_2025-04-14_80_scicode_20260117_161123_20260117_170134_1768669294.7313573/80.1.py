import os
import math
import time
import numpy as np
import scipy as sp
from mpl_toolkits.mplot3d import Axes3D
import pickle
from scipy.constants import  Avogadro

def dist(r1, r2, L):
    dx = r1[0] - r2[0]
    dy = r1[1] - r2[1]
    dz = r1[2] - r2[2]
    dx -= L * round(dx / L)
    dy -= L * round(dy / L)
    dz -= L * round(dz / L)
    distance = (dx*dx + dy*dy + dz*dz) ** 0.5
    return distance

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('80.1', 3)
target = targets[0]

r1 = np.array([2.0, 3.0, 4.0])
r2 = np.array([2.5, 3.5, 4.5])
box_length = 10.0
assert np.allclose(dist(r1, r2, box_length), target)
target = targets[1]

r1 = np.array([1.0, 1.0, 1.0])
r2 = np.array([9.0, 9.0, 9.0])
box_length = 10.0
assert np.allclose(dist(r1, r2, box_length), target)
target = targets[2]

r1 = np.array([0.1, 0.1, 0.1])
r2 = np.array([9.9, 9.9, 9.9])
box_length = 10.0
assert np.allclose(dist(r1, r2, box_length), target)
