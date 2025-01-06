import numpy as np

from interpolation import *



def time_evo(f, params):
    # Map evolution types to their corresponding functions
    evotype_map = {
        'RK1': RK1,
        'RK2': RK2,
        'RK4': RK4,
        'RK5': RK5,
    }
    
    # Check if temporal discretization method exists
    if params.evotype not in evotype_map:
        raise ValueError(f"Invalid evotype: {params.evotype}. Valid options are {list(evotype_map.keys())}.")

    return evotype_map[params.evotype](f, params)
    

def RK1(f, params):
    # Calculate k1
    F = calc_flux(f, params.stencil)
    f_der = (np.roll(F, -1) - F)


    f -= params.v * params.dt / params.dx * f_der
    return f



def RK2(f, params):
    # Calculate k1
    F = calc_flux(f, params.stencil)
    k1 = - params.v * (np.roll(F, -1) - F) / params.dx

    # Calculate k2
    F = calc_flux(f + k1 * params.dt / 2, params.stencil)
    k2 = - params.v * (np.roll(F, -1) - F) / params.dx

    # Calculate the volume averaged values of the new timestep from the k-values
    f += params.dt * k2
    return f


def RK4(f, params):
    # Calculate k1
    F = calc_flux(f, params.stencil)
    k1 = - params.v * (np.roll(F, -1) - F) / params.dx

    # Calculate k2
    F = calc_flux(f + k1 * params.dt / 2, params.stencil)
    k2 = - params.v * (np.roll(F, -1) - F) / params.dx

    # Calculate k3
    F = calc_flux(f + k2 * params.dt / 2, params.stencil)
    k3 = - params.v * (np.roll(F, -1) - F) / params.dx

    # Calculate k4
    F = calc_flux(f + k3 * params.dt, params.stencil)
    k4 = - params.v * (np.roll(F, -1) - F) / params.dx

    # Calculate the volume averaged values of the new timestep from the k-values
    f += params.dt * (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6)
    return f


def RK5(f, params): #Fehlberg's method https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method
    # Calculate k1
    F = calc_flux(f, params.stencil)
    k1 = - params.v * (np.roll(F, -1) - F) / params.dx

    # Calculate k2
    F = calc_flux(f + k1 * params.dt / 4, params.stencil)
    k2 = - params.v * (np.roll(F, -1) - F) / params.dx

    # Calculate k3
    F = calc_flux(f + (3 * k1 + 9 * k2) * params.dt / 32, params.stencil)
    k3 = - params.v * (np.roll(F, -1) - F) / params.dx

    # Calculate k4
    F = calc_flux(f + (1932 * k1 - 7200 * k2 + 7296 * k3) * params.dt / 2197, params.stencil)
    k4 = - params.v * (np.roll(F, -1) - F) / params.dx

    # Calculate k5
    F = calc_flux(f + (k1 * 439 / 216 - k2 * 8 + k3 * 3680 / 513 - k4 * 845 / 4104) * params.dt, params.stencil)
    k5 = - params.v * (np.roll(F, -1) - F) / params.dx

    # Calculate k6
    F = calc_flux(f + (-k1 * 8 / 27 + k2 * 2 - k3 * 3544 / 2565 + k4 * 1859 / 4104 - k5 * 11 / 40) * params.dt, params.stencil)
    k6 = - params.v * (np.roll(F, -1) - F) / params.dx

    # Calculate the volume averaged values of the new timestep from the k-values
    f += params.dt * (k1 * 16 / 135 + k3 * 6656 / 12825 + k4 * 28561 / 56430 - k5 * 9 / 50 + k6 * 2 / 55)
    return f

