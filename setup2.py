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
nt = 1  # Number of iterations to run
nx = 90  # Gridpoints in x direction
ny = 30  # Gridpoints in y direction
C = 0.05  # CFL number
Cx = 0.01  # Artificial viscosity in x direction
Cy = 0.0  # Artificial viscosity in y direction

# """Set extra parameters"""
# pb = 0.6784  # Back pressure for outflow_pb
# Ar = 2  # Area ratio Ae/At for generic Laval nozzle
# Lr = 100  # Length ratio L/yt for generic Laval nozzle
# # Lr = 3
xw, yw = np.load('MoCwall.npy')  # Load wall profile for specified_grid

# """Set initial condition type"""
# IC = ic_des  # Quasi-1D design condition
# # IC = ic_pb  # Quasi-1D solution with shocks

# """Set grid stretching"""
# bx = 0  # x stretch
# # by = -0.5  # y stretch
# by = 0

# bxc = 0  # Converging section
# bxd = 0  # Diverging section

# stretch1 = bxd, bxc, by
# stretch2 = bx, by

# """Set grid type"""
# # GRID = grid_extended(nx, ny, xw, yw, stretch1)  # Minimum length nozzle
# GRID = grid_specified(nx, ny, xw, yw, stretch2)  # Minimum length nozzle
# # GRID = grid_parametric(nx, ny, Ar, Lr, stretch2)  # Generic Laval nozzle

# """Set inlet and outlet boundary conditions."""
# BC = bc_in_sonic, bc_out_float  # Minimum length diverging section
# # BC = bc_in_subsonic, bc_out_float  # Laval nozzle design condition
# # BC = bc_in_stagnant, bc_out_float  # Laval nozzle design condition
# # BC = bc_in_subsonic, bc_out_pb  # Laval nozzle shock capturing
# # BC = bc_in_stagnant, bc_out_pb
# # BC = bc_in_sonic, bc_out_pb

# """Run program"""
# OUTPUT, Ms, RES, i = RUN(nt, GRID, IC, BC, C, Cx, Cy, pb)

# """Extract primitive variables"""
# rho, T, u, v, U = OUTPUT
# V = (u**2 + v**2)**0.5
# M = V/T**0.5
# p = rho*T

# xis, etas, nx, ny, xs, ys, y_x = GRID

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