from termcolor import colored

from initial_distr import FunctionType
from parameters import Parameters
from simulations import *

"""
At this point in the development of the code the options are
func: [Block: 0, Sine: 1]
stencil: UP2, UP3, UP4, UP5, CD2, CD4, P3_3-0, P3_2-1, P3_1-2, WENO1, WENO2
evotype: RK1, RK2, RK4, RK5

Here for example the notation P3_2-1 implies a three point stencil where two point are to the left of the face averaged value 
and one to the right. We always assume the closest points.
If not specified, the stencil can be asumed to be upwind biased.
WENO1 and WENO2 differ only in their smoothness indicator but WENO1 is fourth order while WENO2 is fifth order according to literature:
WENO1: https://www.sciencedirect.com/science/article/pii/S0021999184711879
WENO2: https://www.sciencedirect.com/science/article/pii/S0021999196901308

The file naming convention generally follows the following structure:
STENCIL_TEMPORAL-DISCRETIZATION_(DURATION-OR-TIME)_COURANT-NUMBER_(GRIDSIZE-OR-NORM-OR-ANIMATED).FILETYPE
"""

# Determine some initial parameters of our system
duration = 10
grid_size = 64
exact_grid_size = 1000
C_max = 0.3
xmax, xmin = 1, -1
v = 1
reps = 6

func = FunctionType(1)
stencil = 'WENO1'
evotype = 'RK5'
mult_stencil = ['P3_3-0', 'P3_2-1', 'P3_1-2']

params = Parameters(grid_size, xmin, xmax, duration, v, C_max, evotype, stencil, func, mult_stencil)
print('You are using the ' + colored(f'{func.name}', "magenta") + ' function!! Consider the regularity of this function when plotting the error!')

"""
At this point in the development of the code the following simulations are possible:

- run_comp(params, exact_grid_size, save=False) : shows the final form of numerical simulation superposed on the exact solution

- run_norm_err(params, reps, save=False) : shows the relative error of the L1, L2 and L_infty norms for an increasing grid_size on log-log plot

- run_interp(params, exact_grid_size, save=False) : shows the interpolations of singular points relative to the exact solution

- ani_comp(params, exact_grid_size, save=False) : shows how the numerical solution evolves superposed on the exact solution

- run_multip_interp(params, exact_grid_size, save=False) : shows the interpolations of multiple numerical schemes at once to compare
"""
run_norm_err(params, reps, save=False)

