import numpy as np

floors = 30
floor_height = 3.5
b = 30.0


floors = np.arange(1,floors+1)
z = floors*floor_height

w0 = 1.25
alpha_terrain = 0.16
zg=300.0
k_z = 2.01 * (np.maximum(z,3.0) / zg) ** (2 * alpha_terrain)
