"""
Created on Mon Oct  8 18:10:59 2018

@author: Gabriel Raval, 2018/19

This file contains a set of functions for running a 2D planar CFD simulation
for inviscid nozzle flows using the MacCormack technique.
Edit the lines at the bottom under ###SETUP to change the simulation
parameters.

Designed for use with the Spyder IDE (https://www.spyder-ide.org) and
Python 3.7. Enable the outline pane for quick navigation.
"""

import numpy as np
import gc
import scipy.optimize as opt
from matplotlib import pyplot as plt


def IDstring(nx, ny, i, C, Cx, Cy, pb):
    """Return a list of strings used to identify figures.

    Arguments:
        nx -- Number of x coordinates. (int)\n
        ny -- Number of y coordinates. (int)\n
        i -- Latest iteration number. (int)\n
        C -- Courant number. (float)\n
        Cx -- Amount of artificial viscosity in x direction. (float)\n
        Cy -- Amount of artificial viscosity in y direction. (float)\n
        pb -- Back pressure. (float)\n
    """

    IDs = ['$n_x$=%i' % nx,
           '$n_y$=%i' % ny,
           '$n_t$=%i' % i,
           'C=%s' % f'{C:.2f}',
           '$C_x$=%s' % f'{Cx:.2f}',
           '$C_y$=%s' % f'{Cy:.2f}',
           '$p_b$=%s' % f'{pb:.1f}'
           ]

    return IDs


def plot_mach(OUTPUT, GRID, ID):
    """Return a plot of the nozzle Mach number profile.

    Arguments:
        OUTPUT -- Arrays of data returned by the maccormack function or one of\
                  the ic_* functions. (tuple)

        GRID -- Arrays of data returned by one of the grid_* functions. (tuple)

        ID -- String to add below plot title. (str)
    """

    """Extract grid data"""
    xis, etas, nx, ny, xs, ys, y_x = GRID
    rho, T, u, v, U = OUTPUT
    V = (u**2+v**2)**0.5
    M = V/T**0.5

    f = plt.figure()

    for j in np.arange(ny-3)+1:
        plt.plot(xs[0, :], M[j, :], 'b', linewidth=0.1)

    plt.plot(xs[0, :], M[-2, :], 'b', label='Interior', linewidth=0.1)
    plt.plot(xs[0, :], M[0, :], 'r', label='Centerline', linewidth=1.5)
    plt.plot(xs[0, :], M[-1, :], 'k', label='Wall', linewidth=1.5)

    plt.title("Mach number profile\n%s" % ID)
    plt.xlabel('$x/y_t$')
    plt.xlim(-0.5, xs[0, -1]+0.5)
    plt.xticks(np.linspace(0, int(xs[0, -1]), int(xs[0, -1])+1))

    plt.ylabel('M    ', rotation='horizontal')
    plt.ylim(np.min(M)-0.1, np.max(M)+0.1)

    plt.legend(frameon=False)

    return plt.show(f)


def plot_pressure(OUTPUT, GRID, ID):
    """Return a plot of the nozzle pressure profile.

    Arguments:
        OUTPUT -- Arrays of data returned by the maccormack function or one of\
                  the ic_* functions. (tuple)

        GRID -- Arrays of data returned by one of the grid_* functions. (tuple)

        ID -- String to add below plot title. (str)
    """

    xis, etas, nx, ny, xs, ys, y_x = GRID
    rho, T, u, v, U = OUTPUT
    p = rho*T

    f = plt.figure()
    for j in np.arange(ny-3)+1:
        plt.plot(xs[0, :], p[j, :], 'b', linewidth=0.1)

    plt.plot(xs[0, :], p[-2, :], 'b', label='Interior', linewidth=0.1)
    plt.plot(xs[0, :], p[0, :], 'r', label='Centerline', linewidth=1.5)
    plt.plot(xs[0, :], p[-1, :], 'k', label='Wall', linewidth=1.5)

    plt.title("Pressure profile\n%s" % ID)
    plt.ylabel('p', rotation='horizontal')
    plt.xlabel('$x/y_t$')
    plt.ylim(0, 1)
    plt.legend(frameon=False)

    return plt.show(f)


def plot_mach_time(Ms, i, ID):
    """Return a plot of the outlet centerline Mach number varying over time.
    Reaches a reasonably steady state much sooner than the residuals.

    Arguments:
        Ms -- array of length nt containing outlet Mach numbers over time\
              (numpy.ndarray)

        i -- latest iteration number. (int)

        ID -- String to add below plot title. (str)
    """

    ts = np.arange(i)  # Plot upto latest iteration 'i'
    f = plt.figure()
    plt.plot(ts, Ms[:i, 0])

    plt.title('Outlet centerline Mach number over time\n %s' % ID)
    plt.ylabel('M', rotation='horizontal')
    plt.xlabel('nt')

    return plt.show(f)


def plot_mach_out(OUTPUT, ys, ID):
    """Return a plot of the Mach number profile across the nozzle outlet.

    Arguments:
        OUTPUT -- Arrays of data returned by the maccormack function or one of\
                  the ic_* functions. (tuple)

        ys -- Array of y coordinates for every grid point. (numpy.ndarray)

        ID -- String to add below plot title. (str)
    """

    """Extract data"""
    rho, T, u, v, U = OUTPUT
    V = (u**2+v**2)**0.5
    M = V/T**0.5

    """Create plot"""
    f = plt.figure(figsize=(10, 3))
    plt.plot(ys[:, -1], M[:, -1])

    """Format plot"""
    plt.title("Outlet Mach number profile\n%s" % ID)
    plt.ylabel('M             ', rotation='horizontal')
    plt.ylim(min(M[:, -1]-0.1), max(M[:, -1])+0.1)
    plt.xlabel('$y/y_t$')
    plt.legend(frameon=False)

    return plt.show(f)


def plot_res(RES, i, ID):
    """Returns a plot of the residuals in the style of ANSYS Fluent.

    Arguments:
        RES -- Array returned by RUN function containing residuals for each\
               iteration. (numpy.ndarray)

        i -- Latest iteration number. (int)

        ID -- String to add below plot title. (str)
    """

    ts = np.arange(i)  # Plot upto latest iteration 'i'
    f = plt.figure()

    plt.plot(ts, RES[:i, 0], label="$U_1'$")
    plt.plot(ts, RES[:i, 1], label="$U_2'$")
    plt.plot(ts, RES[:i, 2], label="$U_3'$")
    plt.plot(ts, RES[:i, 3], label="$U_4'$")

    plt.yscale('log')
    plt.legend(frameon=False)
    plt.title("Residuals\n%s" % ID)
    plt.xlabel("nt")

    return plt.show(f)


def plot_grid(GRID, ID):
    """Plots the physical and computational grids.

    Arguments:
        GRID -- Arrays of data returned by one of the grid_* functions. (tuple)

        ID -- String to add below plot title. (str)
    """

    """Extract grid data"""
    xis, etas, nx, ny, xs, ys = GRID[:-1]
    yw = ys[-1]
    scale = 10

    """COMPUTATIONAL GRID"""
    f1 = plt.figure(figsize=(scale, scale))

    """Plot gridlines of constant eta."""
    for j in np.arange(ny):
        plt.plot(xis[j], etas[j], 'b', linewidth=0.75)

    """Plot gridlines of constant xi."""
    plt.plot(xis, etas, 'b', linewidth=0.75)

    """Format plot"""
    plt.axis('scaled')
    plt.xlabel(r'$\xi$')
    plt.ylabel(r'$\eta$', rotation='horizontal')
    plt.xlim(-0.5, xis[0, -1]+0.5)
    plt.ylim(-0.5, etas[-1, 0]+0.5)
    plt.xticks(ticks=np.arange(0, np.ceil(xis[0, -1]), 5))
    plt.yticks(ticks=np.arange(0, np.ceil(etas[-1, -1]), 5))
    plt.title('Computational grid\n%s' % ID)

    """PHYSICAL GRID"""
    f2 = plt.figure(figsize=(scale, scale))

    """Plot lines of constant eta."""
    for j in np.arange(ny):
        plt.plot(xs[j], ys[j], 'b', linewidth=0.75)

    """Plot lines of constant xi."""
    plt.plot(xs, ys, 'b', linewidth=0.75)

    """Plot nozzle wall."""
    plt.plot(xs[0], yw, 'k', linewidth=2)

    """Format plot"""
    plt.axis('scaled')
    plt.xlabel('$x/y_t$')
    plt.ylabel('$y/y_t$        ', rotation='horizontal')
    plt.xticks(ticks=np.arange(0, np.ceil(xs[0, -1]+1), 1))
    plt.yticks(ticks=np.arange(0, np.ceil(ys[-1, -1]+1), 1))
    plt.xlim(-0.5, xs[0, -1]+0.5)
    plt.ylim(-0.5, ys[-1, -1]+0.5)
    plt.title('Physical grid\n%s' % ID)

    return plt.show(f2)


def contour_mach(OUTPUT, GRID, ID):
    """Plots a 2D contour plot of Mach number.

    Arguments:
        OUTPUT -- Arrays of data returned by the maccormack function or one of\
                  the ic_* functions. (tuple)

        GRID -- Arrays of data returned by one of the grid_* functions. (tuple)

        ID -- String to add below plot title. (str)
    """

    """Extract data"""
    rho, T, u, v, U = OUTPUT
    xis, etas, nx, ny, xs, ys, y_x = GRID
    V = (u**2 + v**2)**0.5
    M = V/T**0.5

    """Create contour plot."""
    z1 = np.linspace(0, np.max(M)+0.25, 500)  # List of Mach numbers to colour
    z2 = np.arange(0, np.ceil(np.max(M))+0.25, 0.5)  # Labels on colorbar

    f = plt.figure(figsize=(13, 6))

    plt.contourf(xs, ys, M, z1, cmap='jet')
#    plt.contourf(xs, -ys, M, z1, cmap='jet')  # Show lower half of nozzle

#    plt.contour(xs, ys, M, np.linspace(2.5, 3, 5), colors='k')

    try:  # Skip if colorbar fails due to zero gradient
        plt.colorbar(ticks=z2)
    except ValueError:
        pass

    """Format plot"""
    plt.axis('scaled')
    plt.title("Mach number\n%s" % ID)
    plt.xlabel('$x/y_t$')
    plt.xlim(-0.5, xs[0, -1]+0.5)
    plt.xticks(np.linspace(0, int(xs[0, -1]), int(xs[0, -1])+1))
    plt.ylabel('$y/y_t$        ', rotation='horizontal')
    plt.ylim(-0.5, ys[-1, -1]+0.5)
#    plt.ylim(-ys[-1, -1]-0.5, ys[-1, -1]+0.5)

    return plt.show(f)


def contour_pressure(OUTPUT, GRID, ID):
    """Plots a 2D contour plot of pressure.

    Arguments:
        OUTPUT -- Arrays of data returned by the maccormack function or one of\
                  the ic_* functions. (tuple)

        GRID -- Arrays of data returned by one of the grid_* functions. (tuple)

        ID -- String to add below plot title. (str)
    """

    rho, T, u, v, U = OUTPUT
    xis, etas, nx, ny, xs, ys, y_x = GRID
    p = rho*T

    z1 = np.linspace(0, np.max(p), 500)
    z2 = np.arange(np.floor(np.min(p)), np.ceil(np.max(p)), 0.1)

    f = plt.figure(figsize=(13, 3))

    plt.contourf(xs, ys, p, z1, cmap='jet')

    plt.colorbar(ticks=z2)
    plt.axis('scaled')
    plt.title("Pressure\n%s" % ID)
    plt.xlabel('$x/y_t$')
    plt.xlim(-0.5, xs[0, -1]+0.5)
    plt.xticks(np.linspace(0, int(xs[0, -1]), int(xs[0, -1])+1))
    plt.ylabel('$y/y_t$    ', rotation='horizontal')
    plt.ylim(-0.5, ys[-1, -1]+0.5)

    return plt.show(f)


def contour_C(OUTPUT, GRID, ID):
    """Contour plot of Mach lines.

    Arguments:
        OUTPUT -- Arrays of data returned by the maccormack function or one of\
                  the ic_* functions. (tuple)

        GRID -- Arrays of data returned by one of the grid_* functions. (tuple)

        ID -- String to add below plot title. (str)
    """

    """Extract data"""
    g = 1.4
    rho, T, u, v, U = OUTPUT
    xis, etas, nx, ny, xs, ys, y_x = GRID
    V = (u**2 + v**2)**0.5
    M = V/T**0.5


#    for q in np.arange(ny)[:-1]:
    q = 0
    # Only plot on columns where M > 1
    w = np.where(np.all(M[q:] > 1, 0))[0]

    theta = np.arctan(v/u)[q:, w]  # Flow angle
    nu = (((g+1)/(g-1))**0.5*np.arctan(((g-1)/(g+1)*(M[q:, w]**2-1))**0.5)
          - np.arctan((M[q:, w]**2-1)**0.5))  # Prandtl-Meyer angle

    Rm = nu + theta  # Reimann invariant R+
    Rp = nu - theta  # Reimann invariant R-

    # Range of 10 evenly spaced Reimann invariants
    z = np.linspace(np.min([Rm, Rp]), np.max([Rm, Rp]), 10)

    """Create plot"""
    f = plt.figure()
    # Plot Reimann invariant contours
    plt.contour(xs[q:, w], ys[q:, w], Rm, z, colors='blue', linewidths=1)
    plt.contour(xs[q:, w], ys[q:, w], Rp, z, colors='red', linewidths=1)

    plt.plot(xs[-1, :], ys[-1, :], 'k')  # Plot nozzle wall

    """Format plot"""
    plt.axis('scaled')
    plt.title("Mach lines\n%s" % ID)

    plt.xlabel('$x/y_t$')
    plt.xlim(-0.5, xs[0, -1]+0.5)
    plt.xticks(np.linspace(0, int(xs[0, -1]), int(xs[0, -1])+1))

    plt.ylabel('$y/y_t$        ', rotation='horizontal')
    plt.ylim(-0.5, ys[-1, -1]+0.5)

    plt.legend((plt.plot(-1, -1, '-r', linewidth=1)[0],
                plt.plot(-1, -1, '-b', linewidth=1)[0]), ('$C_+$', '$C_-$'))

    return plt.show(f)


def diff(y, x, a, GRID):
    """Numerically differentiate arrays in vector y with respect to x

    Arguments:
        y -- List of 2D arrays in xi-eta space to differentiate.\
             (numpy.ndarray)

        x -- Differentiate with respect to 'xi' or 'eta'. (str)

        a -- Use 'f' for forward differences, 'r' for rearward differences or\
             'c' for central differences. (str)

        GRID -- Arrays of data returned by one of the grid_* functions. (tuple)
    """

    """Extract grid data."""
    xis, etas, nx, ny = GRID[:4]
    dxi = xis.max()/(nx-1)
    deta = etas.max()/(ny-1)
    dy_dx = np.zeros(np.shape(y))

    """Differentiate using finite differences."""
    if x == 'xi':  # Differentiate with respect to xi
        if a == 'f':  # Use forward differences
            dy_dx[..., :, :-1] = (y[..., :, 1:]-y[..., :, :-1])/dxi

        if a == 'r':  # Use rearward differences
            dy_dx[..., :, 1:] = (y[..., :, 1:]-y[..., :, :-1])/dxi

        if a == 'c':  # Use central differences
            # Interior columns
            dy_dx[..., :, 1:-1] = (y[..., :, 2:]-y[..., :, :-2])/(2*dxi)
            # Left boundary
            dy_dx[..., :, 0] = \
                (-3*y[..., :, 0]+4*y[..., :, 1]-y[..., :, 2])/(2*dxi)
            # Right boundary
            dy_dx[..., :, -1] =\
                (-3*y[..., :, -1]+4*y[..., :, -2]-y[..., :, -3])/(-2*dxi)

    if x == 'eta':  # Differentiate with respect to eta
        if a == 'f':  # Use forward differences
            dy_dx[..., :-1, :] = (y[..., 1:, :]-y[..., :-1, :])/deta

        if a == 'r':  # Use rearward differences
            dy_dx[..., 1:, :] = (y[..., 1:, :]-y[..., :-1, :])/deta

        if a == 'c':  # Use central differences
            # Interior rows
            dy_dx[..., 1:-1, :] = (y[..., 2:, :]-y[..., :-2, :])/(2*deta)
            # Lower boundary
            dy_dx[..., 0, :] =\
                (-3*y[..., 0, :]+4*y[..., 1, :]-y[..., 2, :])/(2*deta)
            # Upper boundary
            dy_dx[..., -1, :] =\
                (-3*y[..., -1, :]+4*y[..., -2, :]-y[..., -3, :])/(-2*deta)

    return dy_dx


def grid_specified(nx, ny, xw, yw, stretch=(0, 0)):
    """Returns a structured grid based on the wall contour specified by xw, yw.

    Arguments:
        nx -- Number of x coordinates. (int)\n
        ny -- Number of y coordinates. (int)\n
        xw -- List of wall x coordinates. (numpy.ndarray)\n
        yw -- List of wall y coordinates. (numpy.ndarray)\n
        stretch -- Tuple containing bx, by, floats for controlling stretch in\
                    x and y direction respectively. (tuple)
    """

    bx, by = stretch

    """Create arrays of computational grid points."""
    xi = np.arange(nx)
    eta = np.arange(ny)
    xis, etas = np.meshgrid(xi, eta)

    """Generate x coordinates."""
    L = xw[-1]  # Nozzle length
    ux = np.linspace(0, 1, nx)  # [0, ..., 1] of length nx

    if bx == 0:
        x = L*ux  # Stretch
    else:
        x = L*(np.e**(bx*ux)-1)/(np.e**bx-1)  # No stretch

    xs = np.tile(x, (ny, 1))  # Final x-coordinate array

    """Generate y coordinates"""
    P = np.polynomial.Polynomial.fit(xw, yw, 10)  # Polynomial fitted to xw, yw
    y = P(x)  # New yw for diverging section
    uy = np.array([eta/eta[-1]]).T
    if by == 0:
        ys = y*uy
    else:
        ys = y*(np.e**(by*uy)-1)/(np.e**by-1)

    """Slope of gridlines (dy/dx)."""
    y_x = np.zeros((ny, nx))
    for j, _ in enumerate(eta):
        y_x[j, :] = np.gradient(ys[j, :], xs[j, :])

    return xis, etas, nx, ny, xs, ys, y_x


def grid_extended(nx, ny, xw, yw, stretch=(0, 0, 0)):
    """Returns a structured grid based on the wall contour specified by xw, yw,
    adds a converging section to make a Laval nozzle.

    Arguments:
        nx -- Number of x coordinates. (int)\n
        ny -- Number of y coordinates. (int)\n
        xw -- List of wall x coordinates. (numpy.ndarray)\n
        yw -- List of wall y coordinates. (numpy.ndarray)\n
        stretch (tuple):
            bxd -- Amount of stretch for diverging section in x direction.\
                   (float)\n
            bxc -- Amount of stretch for converging section in x direction.\
                   (float)\n
            by -- Amount of stretch in y direction. (float)
    """

    bxd, bxc, by = stretch

    """Create empty arrays."""
    xi = np.arange(nx)
    eta = np.arange(ny)
    xis, etas = np.meshgrid(xi, eta)
    y_x, ys = np.zeros((ny, nx)), np.zeros((ny, nx))

    """Generate x coordinates."""
    L = xw[-1]  # Diverging section length
    xt = L/4  # Add converging section of length L/4
#    i_t = round(nx/4)
    w = abs(xt-np.linspace(0, L+xt, nx))
    i_t = np.where(w == np.min(w))[0][0]  # Array index of throat
    ux1 = np.linspace(0, 1, i_t, endpoint=False)  # [0, ..., 1] of length i_t
    ux2 = np.linspace(0, 1, nx-i_t)   # [0, ..., 1] of length nx-i_t

    if bxd == 0:
        x2 = L*ux2  # No stretch
    else:
        x2 = L*(np.e**(bxd*ux2)-1)/(np.e**bxd-1)  # Stretch

    if bxc == 0:
        x1 = xt*ux1  # No stretch
    else:
        x1 = xt*(np.e**(bxc*ux1)-1)/(np.e**bxc-1)  # Stretch

    x = np.append(x1, x2+xt)  # Final x-axis
    xs = np.tile(x, (ny, 1))  # Final x-coordinate array

    """Generate y coordinates"""
    Ar = yw[-1]/yw[0]  # Area ratio A/At
    P = np.polynomial.Polynomial.fit(xw, yw, 10)  # Polynomial fitted to xw, yw
    y2 = P(x2)  # New yw for diverging section
    y1 = -(Ar/2-1)/2*np.cos(np.pi*(xt-x1)/xt)+(Ar/2+1)/2  # Converging section
    # Ensure minimum wall height is at start of diverging section
    y1[-1], y2[0] = np.max([y2[0], y1[-1]]), np.min([y2[0], y1[-1]])

    y = np.array([np.append(y1, y2)])  # Uniform y axis
    uy = np.array([eta/eta[-1]]).T  # [0, ..., 1] of length ny

    if by == 0:
        ys = y*uy  # No stretch
    else:
        ys = y*(np.e**(by*uy)-1)/(np.e**by-1)  # Stretch

    """Slope of gridlines of constant eta on the physical plane."""
    for i, _ in enumerate(eta):
        y_x[i, :] = np.gradient(ys[i, :], xs[i, :])

    return xis, etas, nx, ny, xs, ys, y_x


def grid_parametric(nx, ny, Ar, L, stretch=(0, 0)):
    """Generate a grid for a parametric wall contour.

    Arguments:
        nx -- Number of x coordinates. (int)\n
        ny -- Number of y coordinates. (int)\n
        Ar -- Exit area ratio Ae/At. (float)\n
        L -- Nozzle length. (float)\n
        stretch (tuple):
            bx -- Amount of stretch in x direction. (float)\n
            by -- Amount of stretch in y direction. (float)
    """

    bx, by = stretch

    """Create arrays of xi and eta coordinates on the computational plane."""
    xi = np.arange(nx)
    eta = np.arange(ny)
    xis, etas = np.meshgrid(xi, eta)
    y_x = np.zeros((ny, nx))

    """Create arrays of x and y coordinates on the physical plane base on the
    nozzle area profile."""

    ux = np.linspace(0, 1, nx)

    if bx == 0:
        x = L*ux
    else:
        x = L*(np.e**(bx*ux)-1)/(np.e**bx-1)
    xs = np.tile(x, (ny, 1))


#    "Quadratic wall contour"
#    A = 1 + 2.2*(x-1.5)**2

    "Cosine wall contour with maximum area ratio Ar"
    A = (Ar-1)*np.cos(2*np.pi*x/L)+(Ar+1)

#    "Straight nozzle"
#    A = (1-Ar)*x/L + Ar  # Straight nozzle

    """y-coordinates"""
    y = A/2  # Nozzle wall profile
    uy = np.array([eta/eta[-1]]).T

    if by == 0:
        ys = y*uy
    else:
        ys = y*(np.e**(by*uy)-1)/(np.e**by-1)

    for i, _ in enumerate(eta):
        y_x[i, :] = np.gradient(ys[i, :], xs[i, :])

    return xis, etas, nx, ny, xs, ys, y_x


def ic_pb(GRID, pb, g=1.4):
    """Calculate analytical quasi-1D conditions including shocks.

    Returns the analytical quasi-1D solution vector calculated using isentropic
    flow equations and shock jump relations. Uses the Brent method to solve
    nonlinear equations when necessary.

    Arguments:
        GRID -- Arrays of data returned by one of the grid_* functions. (tuple)

        g -- Ratio of specific heat capacities. (float)

        pb -- Back-pressure. (float)
"""

    """Flow data"""
    pb_p0 = pb  # Back pressure is dimensionless

    """Grid data"""
    xis, etas, nx, ny, xs, ys, y_x = GRID

    """Define arrays for primitive variables and solution vector."""
    rho, T = np.zeros((ny, nx)), np.zeros((ny, nx))  # Density and temperature
    u, v = np.zeros((ny, nx)), np.zeros((ny, nx))  # x and y-velocities
    p = np.zeros((ny, nx))
    V = np.zeros((ny, nx))  # Resultant velocity
    M = np.zeros((ny, nx))  # Mach number
    U = np.zeros((4, ny, nx))  # Solution vector

    """Area profile"""
    A = ys[-1, :]*2  # Double the wall height of a symmetrical nozzle
    At = np.min(A)  # Throat area
    Ar = A/At  # Area ratio A/A*
    Ae = A[-1]  # Exit area
    Ar_e = Ae/At  # Outlet area
    i_t = np.where(A == At)[0].max()  # Throat location

    def F_Ar(M, g):
        "Nozzle area A/A* as a function of Mach number."
        return 1/M*((1+(g-1)/2*M**2)/(1+(g-1)/2))**((g+1)/(2*g-2))

    def M_sub(Ar, g):
        """Takes an area ratio A/A* and returns the corresponding subsonic Mach
        number using a variant of the Brent root finding method."""
        if np.any(Ar < 1):
            raise ValueError("A/A* must be greater than or equal to 1")
        else:
            return opt.brenth(lambda M: F_Ar(M, g) - Ar, 10**-12, 1)

    def M_super(Ar, g):
        """Takes an area ratio A/A* and returns the corresponding supersonic
        Mach number using a variant of the Brent root finding method."""
        if np.any(Ar < 1):
            raise ValueError("A/A* must be greater than or equal to 1")
        else:
            try:
                return opt.brenth(lambda M: F_Ar(M, g) - Ar, 1, 5)
            except ValueError:
                raise ValueError("Invalid above Mach 5")

    def p_p0(M, g):
        "Pressure p/p0 as a function of Mach number."
        return (1 + (g-1)/2*M**2)**(-g/(g-1))

    def p02_p01(M, g):
        """Total pressure ratio p02/p01 across a normal shock as a function of
        Mach number M1 immediately before shock."""
        ans = ((g+1)*M**2/((g-1)*M**2+2))**(g/(g-1))\
            * ((g+1)/(2*g*M**2-g+1))**(1/(g-1))
        return ans

    "Design exit Mach number."
    Me_D = M_super(Ar_e, g)

    "Minimum choked exit Mach number (sonic only at throat)."
    Me_C = M_sub(Ar_e, g)

    "Design exit pressure pe/p0."
    pe_p0_D = p_p0(Me_D, g)

    "Maximum choked exit pressure pe/p0 (sonic only at throat)."
    pe_p0_C = p_p0(Me_C, g)

    if pb_p0 == 1:
        "Reservoir condition"
        M[:, :] = 0

    elif pb_p0 > pe_p0_C:
        "Everywhere subsonic"
        Me = ((pb_p0**((1-g)/g) - 1)*2/(g-1))**0.5
        A1_sonic = Ae/F_Ar(Me, g)
        M[:, :] = np.array([M_sub(A[i]/A1_sonic, g) for i in np.arange(nx)])

    else:
        "All other exit pressures will have sonic throat"
        M[:, :i_t] = np.array([M_sub(Ar[i], g) for i in np.arange(i_t)])

    if pb_p0 <= pe_p0_D:
        "Either in design condition or underexpanded."
        M[:, i_t:] = np.array([M_super(Ar[i], g) for i in np.arange(i_t, nx)])

    elif pb_p0 < pe_p0_C:
        "Either shock in nozzle or overexpanded."
        try:
            "Shock in nozzle"
            f1 = lambda M: F_Ar(M, g)*p_p0(M, g) - pb_p0*Ar_e
            Me = opt.brenth(f1, Me_C, 1)  # Look for Me_C < Me < 1

            "Mach number M1 immediately before shock."
            pb_p02 = p_p0(Me, g)
            f2 = lambda M: p02_p01(M, g) - pb_p0/pb_p02
            M1 = opt.brenth(f2, 1, Me_D)  # Look for 1 < M1 < Me_D

            "Mach number M2 immediately after shock."
            M2 = ((1 + (g-1)/2 * M1**2) / (g*M1**2 - (g-1)/2))**0.5

            "Area ratio As/At = As/A1* at location of shock."
            Ar1 = F_Ar(M1, g)

            "Area ratio As/A2* at location of shock."
            Ar2 = F_Ar(M2, g)

            "Sonic throat area A2* after normal shock."
            A2sonic = Ar1*At/Ar2

            "Array index of normal shock."
            i_s = np.where(Ar <= Ar1)[0].max() + 2  # Array index of shock

            "Supersonic Mach number between throat and normal shock."
            M[:, i_t:i_s] = [M_super(A[i]/At, g) for i in np.arange(i_t, i_s)]

            "Subsonic Mach number between normal shock and exit."
            M[:, i_s:] = [M_sub(A[i]/A2sonic, g) for i in np.arange(i_s, nx)]

        except ValueError:
            "Overexpanded"
            M[:, i_t:] = np.array([M_super(Ar[i], g)
                                  for i in np.arange(i_t, nx)])

    T = (1 + (g-1)/2*M**2)**-1  # Stagnation temperature unchanged
    try:
        p[:, :i_s] = T[:, :i_s]**(g/(g-1))  # Different stagnation pressures
        p[:, i_s:] = T[:, i_s:]**(g/(g-1))*p02_p01(M1, g)
    except UnboundLocalError:
        p = T**(g/(g-1))
    rho[:, :] = p/T
    V = M*T**0.5

    theta = np.arctan(y_x)  # Angle of lines of constant eta
    u = V*np.cos(theta)  # Correct x-velocity
    v = V*np.sin(theta)  # Correct y-velocity

    """Initial conditions of solution vector."""
    U[0] = rho
    U[1] = rho*u
    U[2] = rho*v
    U[3] = rho*(T/(g**2-g) + 0.5*V**2)

    return rho, T, u, v, U


def ic_des(GRID, g=1.4):
    """Returns the isentropic quasi-1D solution using the ic_pb function.

    Arguments:
        GRID -- Arrays of data returned by one of the grid_* functions. (tuple)

        g -- Ratio of specific heat capacities. (float)
    """

    return ic_pb(GRID, 0, g)


def timestep(GRID, u, v, T, C):
    """Calculates the longest stable timestep for an isentropic flow.

    Arguments:
        GRID -- Arrays of data returned by one of the grid_* functions. (tuple)

        u -- Velocities in x direction. (numpy.ndarray)\n
        v -- Velocities in y direction. (numpy.ndarray)\n
        T -- Temperatures. (numpy.ndarray)\n
        C -- Courant number for scaling the timestep. Must be < 1. (float)
    """

    """Grid data"""
    xis, etas, nx, ny, xs, ys, y_x = GRID

    """Smallest distances between gridpoints in x and y directions."""
    dx = abs(xs[:, 1:] - xs[:, :-1]).min()
    dy = abs(ys[1:, :] - ys[:-1, :]).min()

    """Calculate a value of dt for each gridpoint."""
    dt = C/(np.abs(u + T**0.5)/dx + np.abs(v + T**0.5)/dy)

    """Choose smallest dt."""
    dt = np.min(dt)

    return dt


def metrics(GRID):
    """Calculates the inverse metrics for given grid data.

    Arguments:
        GRID -- Arrays of data returned by one of the grid_* functions. (tuple)
    """

    """Grid data"""
    xis, etas, nx, ny, xs, ys = GRID[:-1]

    """Inverse metrics"""
    y_xi = diff(ys, 'xi', 'c', GRID)
    y_eta = diff(ys, 'eta', 'c', GRID)
    x_xi = diff(xs, 'xi', 'c', GRID)
    x_eta = diff(xs, 'eta', 'c', GRID)

    return y_xi, y_eta, x_xi, x_eta


def euler(U, GRID, g=1.4):
    """Returns flux vectors for the Euler equations.

    Arguments:
        U -- Solution vector U or Up from the maccormack function.\
             (numpy.ndarray)

        GRID -- Arrays of data returned by one of the grid_* functions. (tuple)

        g -- Ratio of specific heat capacities. (float)
    """

    """Grid data"""
    xis, etas, nx, ny, xs, ys, y_x = GRID

    """Define flux vectors in terms of solution vector."""
    F = np.zeros((4, ny, nx))
    F[0] = U[1]
    F[1] = U[1]**2/U[0] + (g-1)*(U[3]-(U[1]**2+U[2]**2)/(2*U[0]))
    F[2] = (U[1]*U[2])/U[0]
    F[3] = U[1]*(g*U[3]/U[0] - (g-1)*(U[1]**2+U[2]**2)/(2*U[0]**2))

    G = np.zeros((4, ny, nx))
    G[0] = U[2]
    G[1] = (U[1]*U[2])/U[0]
    G[2] = U[2]**2/U[0] + (g-1)*(U[3]-(U[1]**2+U[2]**2)/(2*U[0]))
    G[3] = U[2]*(g*U[3]/U[0] - (g-1)*(U[1]**2+U[2]**2)/(2*U[0]**2))

    return F, G


def av(U, nx, ny):
    """Returns artificial viscosity terms Sx, Sy.

    Arguments:
        U -- Solution vector U or Up from the maccormack function.\
             (numpy.ndarray)\n
        nx -- Number of x coordinates. (int)\n
        ny -- Number of y coordinates. (int)
    """

    rho = U[0]
    u = U[1]/U[0]
    v = U[2]/U[0]
    V = (u**2 + v**2)**0.5
    T = (g**2-g)*(U[3]/U[0] - 0.5*V**2)
    p = rho*T

    Sx, Sy = np.zeros((4, ny, nx)), np.zeros((4, ny, nx))
    Sx[:, :, 1:-1] = Cx*np.abs(p[:, 2:] - 2*p[:, 1:-1] + p[:, :-2])\
        / (p[:, 2:] + 2*p[:, 1:-1] + p[:, :-2])\
        * (U[:, :, 2:] - 2*U[:, :, 1: -1] + U[:, :, :-2])

    Sy[:, 1:-1, :] = Cy*np.abs(p[2:, :] - 2*p[1:-1, :] + p[:-2, :])\
        / (p[2:, :] + 2*p[1:-1, :] + p[:-2, :])\
        * (U[:, 2:, :] - 2*U[:, 1: -1, :] + U[:, :-2, :])

    return Sx, Sy


def maccormack(U, GRID, MET, C, Cx, Cy, pb, i, g=1.4):
    """Executes one iteration of the MacCormack time marching method.

    Arguments:
        U -- Solution vector from the maccormack function or one of the ic_*\
             functions. (numpy.ndarray)

        GRID -- Arrays of data returned by one of the grid_* functions. (tuple)

        MET -- Arrays of inverse metrics returned by the metrics function.\
               (tuple)

        C -- Courant number. (float)\n
        Cx -- Amount of artificial viscosity in x direction. (float)\n
        Cy -- Amount of artificial viscosity in y direction. (float)\n
        pb -- Back pressure. (float)\n
        i -- Current iteration number from RUN function. (int)\n
        g -- Ratio of specific heat capacities. (float)
    """

    """Calculate timestep for the given CFL number."""
    u = U[1]/U[0]
    v = U[2]/U[0]
    V = (u**2 + v**2)**0.5
    T = (g**2-g)*(U[3]/U[0] - 0.5*V**2)
    dt = timestep(GRID, u, v, T, C)

    """Switch order of forward and rearward differences each timestep."""
    # Even iteration
    if i % 2 == 0:
        a = 'f'  # Forward differences in the predictor step
        b = 'r'  # Rearward differences in the corrector step
    # Odd iteration
    else:
        a = 'r'  # Rearward differences in the predictor step
        b = 'f'  # Forward differences in the corrector step

    """Grid data and inverse metrics."""
    xis, etas, nx, ny, xs, ys, y_x = GRID
    y_xi, y_eta, x_xi, x_eta = MET

    """Transform solution and flux vectors to xi-eta coordinates."""
    F, G = euler(U, GRID, g)
    J = x_xi*y_eta - x_eta*y_xi  # Jacobian determinant from inverse metrics

    U1 = J*U
    F1 = F*y_eta - G*x_eta
    G1 = -F*y_xi + G*x_xi

    """Predictor step (time t)"""
    F1_xi = diff(F1, 'xi', a, GRID)
    G1_eta = diff(G1, 'eta', a, GRID)
    U1_t = -F1_xi - G1_eta

    """Artificial viscosity"""
    Sx, Sy = av(U, nx, ny)

    """Predicted solution vector (time t+dt)."""
    U1p = U1 + U1_t*dt + (Sx + Sy)*J  # 'p' refers to predicted value

    """Transform predicted solution vector and calculate flux vectors."""
    Up = U1p/J
    Fp, Gp = euler(Up, GRID, g)

    """Transform flux vectors."""
    F1p = Fp*y_eta - Gp*x_eta
    G1p = -Fp*y_xi + Gp*x_xi

    """Corrector step"""
    F1p_xi = diff(F1p, 'xi', b, GRID)
    G1p_eta = diff(G1p, 'eta', b, GRID)
    U1p_t = -F1p_xi - G1p_eta
    U1_t_avg = 0.5*(U1_t + U1p_t)

    """Artificial viscosity"""
    Sxp, Syp = av(Up, nx, ny)

    """Corrected solution vector."""
    U1 = U1 + U1_t_avg*dt + (Sxp + Syp)*J

    """Transform solution vector back to x-y coordinates."""
    U = U1/J

    """Boundary conditions"""
    if bc_in_sonic in BC:
        U_in = bc_in_sonic(ny, g)
    elif bc_in_subsonic in BC:
        U_in = bc_in_subsonic(U, ny, y_x, g)
    elif bc_in_stagnant in BC:
        U_in = bc_in_stagnant(U, ny, y_x, g)

    if bc_out_pb in BC:
        U_out = bc_out_pb(U, ny, y_x, pb, g)
    elif bc_out_float in BC:
        U_out = bc_out_float(U)

    U_wall = bc_wall(U, y_x)
    U_CL = bc_sym(U)

    U[:, :, -1] = U_out
    U[:, -1, :] = U_wall
    U[:, 0, :] = U_CL
    U[:, :, 0] = U_in

    """Update primitive variables"""
    rho = U[0]
    u = U[1]/U[0]
    v = U[2]/U[0]
    V = (u**2 + v**2)**0.5
    T = (g**2-g)*(U[3]/rho - 0.5*V**2)
#    print(1, np.max(abs()))
    return rho, T, u, v, U


def bc_in_sonic(ny, g=1.4):
    """Sonic inlet condition.

    Uses isentropic flow relations to calculate temperature, velocity and
    density for a sonic nozzle inlet and returns the corresponding 4 by ny
    U vector for the inlet.

    Arguments:
        ny -- Number of y coordinates. (int)\n
        g -- Ratio of specific heat capacities. (float)
    """

    """Define empty arrays."""
    U_in = np.zeros((4, ny))
    T_in, V_in, rho_in = np.zeros(ny), np.zeros(ny), np.zeros(ny)

    """Calculate flow variables at Mach 1."""
    T_in = (1 + (g-1)*1.0**2/2)**-1  # T/T0 at Mach 1
    V_in = T_in**0.5  # V = a0 at Mach 1
    rho_in = T_in**(1/(g-1))  # rho/rho0 at Mach 1
    u_in = V_in  # Straight, uniform inflow
    v_in = 0

    """U vector for inlet gridpoints."""
    U_in[0] = rho_in
    U_in[1] = rho_in*u_in
    U_in[2] = rho_in*v_in
    U_in[3] = rho_in*(T_in/(g**2-g) + V_in**2/2)

    return U_in


def bc_in_subsonic(U, ny, y_x, g=1.4):
    """Floating inlet condition.

    Generates 4 by ny vector U at the inlet by extrapolating velocity and
    calculating temperature and density from isentropic flow relations.

    Arguments:
        U -- Solution vector from the maccormack function. (numpy.ndarray)\n
        ny -- Number of y coordinates. (int)\n
        y_x -- Gradient of y coordinates in x direction. (np.ndarray)\n
        g -- Ratio of specific heat capacities. (float)
    """

#    alpha_in = np.arctan(y_x)[:, 0]  # Angle of eta-lines at inflow
    alpha_in = 0  # Uncomment this to make inflow uniform

    U_in = np.zeros((4, ny))  # Define empty array
    U_in[:-1, :] = 2*U[:-1, :, 1] - U[:-1, :, 2]  # Extrapolate U1, U2 and U3
    V_in = (U_in[1, :]**2 + U_in[2, :]**2)**0.5/U_in[0, :]
    u_in = V_in*np.cos(alpha_in)  # Correct x-velocity
    v_in = V_in*np.sin(alpha_in)  # Correct y-velocity
    T_in = 1 - ((g-1)/2)*V_in**2  # Isentropic flow relation
    rho_in = T_in**(1/(g-1))  # Isentropic flow relation

    """U vector for inlet gridpoints."""
    U_in[0, :] = rho_in
    U_in[1, :] = rho_in*u_in
    U_in[2, :] = rho_in*v_in
    U_in[3, :] = rho_in*(T_in/(g**2-g) + V_in**2/2)

    return U_in


def bc_in_stagnant(U, ny, y_x, g=1.4):
    """Stagnation inlet condition.

    Generates 4 by ny vector U at the inlet by extrapolating velocity and
    setting temperature and density to their stagnation values.

    Arguments:
        U -- Solution vector from the maccormack function. (numpy.ndarray)\n
        ny -- Number of y coordinates. (int)\n
        y_x -- Gradient of y coordinates in x direction. (np.ndarray)\n
        g -- Ratio of specific heat capacities. (float)
    """

#    alpha_in = np.arctan(y_x)[:, 0]  # Angle of eta-lines at inflow
    alpha_in = 0  # Uncomment this to make inflow uniform

    T_in = 1  # Stagnation value
    rho_in = 1  # Stagnation value

    U_in = np.zeros((4, ny))  # Define empty array
    U_in[:-1, :] = 2*U[:-1, :, 1] - U[:-1, :, 2]  # Extrapolate U1, U2 and U3
    V_in = (U_in[1, :]**2 + U_in[2, :]**2)**0.5/U_in[0, :]
    u_in = V_in*np.cos(alpha_in)  # Correct x-velocity
    v_in = V_in*np.sin(alpha_in)  # Correct y-velocity

    """U vector for inlet gridpoints."""
    U_in[0, :] = rho_in
    U_in[1, :] = rho_in*u_in
    U_in[2, :] = rho_in*v_in
    U_in[3, :] = rho_in*(T_in/(g**2-g) + V_in**2/2)

    return U_in


def bc_out_pb(U, ny, y_x, pb, g=1.4):
    """Outlet back-pressure pb is specified, the rest of the flow variables
    are extrapolated.

    Arguments:
        U -- Solution vector from the maccormack function. (numpy.ndarray)\n
        ny -- Number of y coordinates. (int)\n
        y_x -- Gradient of y coordinates in x direction. (np.ndarray)\n
        pb -- Back pressure. (float)\n
        g -- Ratio of specific heat capacities. (float)
    """

    U_o = np.zeros((4, ny))  # Define empty array
    U_o[:-1, :] = 2*U[:-1, :, -2] - U[:-1, :, -3]  # Extrapolate U1, U2 and U3
    rho_o = U_o[0, :]  # Density at outflow
    V_o = (U_o[1, :]**2 + U_o[2, :]**2)**0.5/rho_o  # Velocity at outflow

    "Calculate U4 at outflow for specified back pressure."
    U_o[3, :] = pb/(g**2-g) + rho_o*V_o**2/2

    return U_o


def bc_out_float(U):
    """Floating outlet condition.

    Calculates U vector at the outlet by extrapolating from previous
    gridpoints. Returns a 4 by ny array.

    Arguments:
        U -- Solution vector from the maccormack function. (numpy.ndarray)
    """

    U_o = 2*U[:, :, -2] - U[:, :, -3]

    return U_o


def bc_sym(U):
    """Symmetry boundary condition.

    Takes the flow variables along the symmetry
    line to be equal to the adjacent gridpoints. Adjusts the flow velocity
    components so the flow is tangential to the symmetry line.
    Returns a 4 by nx array.

    Arguments:
        U -- Solution vector from the maccormack function. (numpy.ndarray)
    """

    U[:, 0, :] = U[:, 1, :]  # Copy flow variables from adjacent gridpoints
    U_s = U[:, 0, :]

    rho_s = U_s[0, :]  # Density
    V_s = (U_s[1, :]**2 + U_s[2, :]**2)**0.5/rho_s  # Velocity
    u_s = V_s  # Correct x-velocity
    v_s = 0  # Correct y-velocity

    "Update U2 and U3 at symmetry line."
    U[1, 0, :] = rho_s*u_s
    U[2, 0, :] = rho_s*v_s

    return U[:, 0, :]


def bc_wall(U, y_x):
    """Inviscid wall boundary condition.

    Extrapolates flow variables along wall
    and adjusts velocity components so the flow is tangential to the wall.
    Returns a 4 by nx array.

    Arguments:
        U -- Solution vector from the maccormack function. (numpy.ndarray)\n
        y_x -- Gradient of y coordinates in x direction. (np.ndarray)
    """

    U_w = 2*U[:, -2, :] - U[:, -3, :]  # Extrapolate U
    alpha_w = np.arctan(y_x)[-1, :]  # Angle of wall
    rho_w = U_w[0, :]  # Density at wall
    V_w = (U_w[1, :]**2 + U_w[2, :]**2)**0.5/rho_w  # Velocity at wall
    u_w = V_w*np.cos(alpha_w)  # Correct x-velocity
    v_w = V_w*np.sin(alpha_w)  # Correct y-velocity

    "Update U2 and U3 at wall."
    U_w[1, :] = rho_w*u_w
    U_w[2, :] = rho_w*v_w

    return U_w


def RUN(nt, GRID, IC, BC, C, Cx, Cy, pb, g=1.4):
    """Returns the results of nt iterations of the maccormack function,
    stopping early if converged.

    Arguments:
        nt -- number of iterations to run. (int)\n
        GRID -- Arrays of data returned by one of the grid_* functions. (tuple)

        IC -- An ic_* initial condition function. (function)\n
        BC -- A pair of bc_in_*/bc_out_* boundary condition functions.\
              (tuple)\n
        C -- Courant number for scaling the timestep. Must be < 1. (float)\n
        Cx -- Amount of artificial viscosity in x direction. (float)\n
        Cy -- Amount of artificial viscosity in y direction. (float)\n
        pb -- Back pressure. (float)\n
        g -- Ratio of specific heat capacities. (float)
    """

    plt.close('all')
    """Load wall coordinates, generate physical and computational grids for
    given nx and ny."""

    xis, etas, nx, ny, xs, ys, y_x = GRID

    """Calculate inverse metrics."""
    MET = metrics(GRID)

    """Calculate initial conditions from quasi-1D exact solution."""
    if IC == ic_pb:
        OUTPUT = IC(GRID, pb, g)
    elif IC == ic_des:
        OUTPUT = ic_des(GRID, g)

    rho, T, u, v, U = OUTPUT

#    """Calculate timestep for the given CFL number."""
#    dt = timestep(GRID, u, v, T, C)

    """Create array for residuals and outlet Mach number at each timestep."""
    RES = np.zeros((nt, 4))
    Ms = np.zeros((nt, ny))

    """Stop iterating if residuals drop below these values."""
    Cs = 10**-6
    C1 = Cs
    C2 = Cs
    C3 = Cs
    C4 = Cs

    d = 15

    """Use initial conditions for first odd and even timestep solutions."""
    PREV1, PREV2 = U, U

    """Loop maccormack() over nt iterations."""
    i = 0
    for i in np.arange(nt):
        try:
            OUTPUT = maccormack(U, GRID, MET, C, Cx, Cy, pb, i, g)
            rho, T, u, v, U = OUTPUT
#            dt = timestep(GRID, u, v, T, C)
            """Residual array at even timesteps is the absolute difference
            between current solution vector and the solution vector from the
            last even timestep."""
            # Even iteration
            if i % 2 == 0:
                RES[i] = abs(U - PREV2).mean((1, 2))*(nx*ny)**0.5
                PREV2 = U  # Previous even-iteration solution vector.
            # Odd iteration
            if i % 2 != 0:
                RES[i] = abs(U - PREV1).mean((1, 2))*(nx*ny)**0.5
                PREV1 = U  # Previous odd-iteration solution vector.

            "Print the mean residual for U1, U2, and U3 to d decimal places."
            R = RES[0]
            if i == 0:
                print(("{:%is}{:%is}{:%is}{:%is}{:s}"
                       % (d, d, d, d))
                      .format('U1', 'U2', 'U3', 'U4', 'iter'))
            if i == 0 or (i+1) % int(nt/10) == 0:  # Print every 10th iteration
                R = RES[i]
                print(("{:<%ie}{:<%ie}{:<%ie}{:<%ie}{:>%is}"
                       % (d, d, d, d, 2*len(str(nt))+1))
                      .format(R[0], R[1], R[2], R[3], "%i/%i" % (i+1, nt)))

            """Stop the loop and save data if the solution blows up."""
            if np.any(np.isnan(U)):
                """ID string for the last iteration."""
                ID = IDstring(nx, ny, i+1, C, Cx, Cy, pb)
                plot_res(RES, i, ID)
                raise RuntimeError('NaN found (%i loops)' % (i+1))

            elif np.all(RES[i] < [C1, C2, C3, C4]):
                print('Converged after %i loops' % (i+1))
                break

            else:
                pass

            Ms[i] = ((u[:, -1]**2+v[:, -1]**2)**0.5/T[:, -1]**0.5)

        except KeyboardInterrupt:
            "Allow loop to be stopped at any time, displaying results."
            print("Interrupted by user")
            break

    gc.collect()

    return OUTPUT, Ms, RES, i


### SETUP
"""Set parameters"""
g = 1.4  # Heat capacity ratio for gas
nt = 0  # Number of iterations to run
nx = 160 # Gridpoints in x direction
ny = 10  # Gridpoints in y direction
C = 0.01  # CFL number
Cx = 0.005  # Artificial viscosity in x direction
Cy = 0.0  # Artificial viscosity in y direction

"""Set extra parameters"""
pb = 0.6784  # Back pressure for outflow_pb
Ar = 2  # Area ratio Ae/At for generic Laval nozzle
Lr = 100  # Length ratio L/yt for generic Laval nozzle
#Lr = 3
xw, yw = np.load('MoCwall.npy')  # Load wall profile for specified_grid

"""Set initial condition type"""
#IC = ic_des # Quasi-1D design condition
IC = ic_pb  # Quasi-1D solution with shocks

"""Set grid stretching"""
bx = 0  # x stretch
#by = -0.5  # y stretch
by = 0

bxc = 0 # Converging section
bxd = 0  # Diverging section

stretch1 = bxd, bxc, by
stretch2 = bx, by

"""Set grid type"""
#GRID = grid_extended(nx, ny, xw, yw, stretch1)  # Minimum length nozzle
#GRID = grid_specified(nx, ny, xw, yw, stretch2)  # Minimum length nozzle
GRID = grid_parametric(nx, ny, Ar, Lr, stretch2)  # Generic Laval nozzle

"""Set inlet and outlet boundary conditions."""
#BC = bc_in_sonic, bc_out_float  # Minimum length diverging section
#BC = bc_in_subsonic, bc_out_float  # Laval nozzle design condition
#BC = bc_in_stagnant, bc_out_float  # Laval nozzle design condition
BC = bc_in_subsonic, bc_out_pb  # Laval nozzle shock capturing
#BC = bc_in_stagnant, bc_out_pb
#BC = bc_in_sonic, bc_out_pb

"""Run program"""
OUTPUT, Ms, RES, i = RUN(nt, GRID, IC, BC, C, Cx, Cy, pb)

"""Extract primitive variables"""
rho, T, u, v, U = OUTPUT
V = (u**2 + v**2)**0.5
M = V/T**0.5
p = rho*T

xis, etas, nx, ny, xs, ys, y_x = GRID

### PLOTS
"""Create ID string for identifying figures."""
IDs = IDstring(nx, ny, i+1, C, Cx, Cy, pb)  # Create list of strings for ID
ID = '  '.join(IDs[:-1])  # don't show pb
#ID = '  '.join(IDs[:2])  # Show nx, ny only
#ID += '\n'
#ID += '  $p_b$=%s' % f'{pb:.1f}'
#ID += '  $\\beta_x$=%s  $\\beta_y$=%s' % (f'{bx:.1f}', f'{by:.1f}')
#ID += '  $\\beta$$_x$$_c$=%s  $\\beta$$_x$$_d$=%s  $\\beta_y$=%s'\
#    % (f'{bxc:.1f}', f'{bxd:.1f}', f'{by:.1f}')

"""Uncomment desired result plots."""
#plot_grid(GRID, ID)

#contour_C(OUTPUT, GRID, ID)
contour_mach(OUTPUT, GRID, ID)
##contour_pressure(OUTPUT, GRID, ID)
#plot_pressure(OUTPUT, GRID, ID)
plot_mach(OUTPUT, GRID, ID)
#plot_mach_out(OUTPUT, ys, ID)
plot_mach_time(Ms, i, ID)
plot_res(RES, i, ID)
