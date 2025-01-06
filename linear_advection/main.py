from initial_distr import FunctionType
from parameters import Parameters
from simulations import *

# Determine some initial parameters of our system
duration = 1
grid_size = 64
exact_grid_size = 1000
C_max = 0.3
xmax, xmin = 1, -1
v = 1
reps = 6

"""
At this point in the development of the code the options are
func: [Block: 0, Sine: 1]
stencil: UP2, UP3, UP4, UP5, CD2, CD4, P3_3-0, P3_2-1, P3_1-2, WENO1
evotype: RK1, RK2, RK5

Here for example the notation P3_2-1 imples a three point stencil where two point are to the left of the face averaged value 
and one to the right. We always assume the closest points.
If not specified, the stencil can be asumed to be upwind biased.
"""
func = FunctionType(0)
stencil = 'UP2'
evotype = 'RK2'
mult_stencil = ['P3_3-0', 'P3_2-1', 'P3_1-2']

params = Parameters(grid_size, xmin, xmax, duration, v, C_max, evotype, stencil, func, mult_stencil)




"""
At this point in the development of the code the following simulations are possible:

- run_comp(params, exact_grid_size, save=False) : shows the final form of numerical simulation superposed on the exact solution

- run_norm_err(params, reps, save=False) : shows the relative error of the L1, L2 and L_infty norms for an increasing grid_size on log-log plot

- run_interp(params, exact_grid_size, save=False) : shows the interpolations of singular points relative to the exact solution

- ani_comp(params, exact_grid_size, save=False) : shows how the numerical solution evolves superposed on the exact solution

- run_multip_interp(params, exact_grid_size, save=False) : shows the interpolations of multiple numerical schemes at once to compare
"""
run_norm_err(params, reps, save=False)

