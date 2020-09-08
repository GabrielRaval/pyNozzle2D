#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 08:56:47 2020

@author: gabriel
"""
import numpy as np

### SETUP
"""Set parameters"""
g = 1.4  # Heat capacity ratio for gas
nt = 10000  # Number of iterations to run
nx = 60  # Gridpoints in x direction
ny = 20  # Gridpoints in y direction
C = 0.05  # CFL number
Cx = 0.05  # Artificial viscosity in x direction
Cy = 0.0  # Artificial viscosity in y direction

# """Set extra parameters"""
# pb = 0.6784  # Back pressure for outflow_pb
Ar = 2  # Area ratio Ae/At for generic Laval nozzle
# Lr = 100  # Length ratio L/yt for generic Laval nozzle
Lr = 10
xw, yw = np.load('MoCwall.npy')  # Load wall profile for specified_grid

# """Set grid stretching"""
# bx = 0  # x stretch
# # by = -0.5  # y stretch
# by = 0

# bxc = 0  # Converging section
# bxd = 0  # Diverging section

# stretch1 = bxd, bxc, by
# stretch2 = bx, by

"""Set grid type"""
# grid = 'specified'
grid = 'extended'
# grid = 'parametric'


"""Set inlet and outlet boundary conditions."""
# inlet = 'sonic'
inlet = 'subsonic'
# inlet = 'stagnant'

# outlet = 'specified'
outlet = 'float'


# ### PLOTS
# """Create ID string for identifying figures."""
# IDs = IDstring(nx, ny, i+1, C, Cx, Cy, pb)  # Create list of strings for ID
# ID = '  '.join(IDs[:-1])  # don't show pb
# # ID = '  '.join(IDs[:2])  # Show nx, ny only
# # ID += '\n'
# ID += f'  $p_b$={pb:.1f}'
# # ID += '  $\\beta_x$=%s  $\\beta_y$=%s' % (f'{bx:.1f}', f'{by:.1f}')
# # ID += '  $\\beta$$_x$$_c$=%s  $\\beta$$_x$$_d$=%s  $\\beta_y$=%s'\
# #     % (f'{bxc:.1f}', f'{bxd:.1f}', f'{by:.1f}')

# """Uncomment desired result plots."""
# # plot_grid(GRID, ID)

# # contour_C(OUTPUT, GRID, ID)
# contour_mach(OUTPUT, GRID, ID)
# # contour_pressure(OUTPUT, GRID, ID)
# # plot_pressure(OUTPUT, GRID, ID)
# # plot_mach(OUTPUT, GRID, ID)
# # plot_mach_out(OUTPUT, ys, ID)
# # plot_mach_time(Ms, i, ID)
# # plot_res(RES, i, ID)