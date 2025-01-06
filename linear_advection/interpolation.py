import numpy as np


def calc_flux(f, stencil: str):
    # Map stencil names to their corresponding functions
    stencil_map = {
        'UP2': UP2,
        'UP3': UP3,
        'UP4': UP4,
        'UP5': UP5,
        'CD2': CD2,
        'CD4': CD4,
        'P3_3-0': P3_3,
        'P3_2-1': UP3,
        'P3_1-2': DOWN3,
        'WENO1': WENO1,

    }
    
    # Check if stencil exists
    if stencil not in stencil_map:
        raise ValueError(f"Invalid stencil option: {stencil}. Valid options are {list(stencil_map.keys())}.")
    
    return stencil_map[stencil](f)


def UP2(f):
    F = (3 * np.roll(f, 1) - np.roll(f, 2)) / 2
    return F


def UP3(f):
    F = (2 * f + 5 * np.roll(f,1) - np.roll(f, 2)) / 6
    return F


def UP4(f): 
    F = (np.roll(f, 3) - 5 *  np.roll(f, 2) + 13 * np.roll(f, 1) + 3 * f) / 12
    return F


def UP5(f):
    F = (2 * np.roll(f, 3) - 13 *  np.roll(f, 2) + 47 * np.roll(f, 1) + 27 * f - 3 * np.roll(f, -1)) / 12
    return F


def CD2(f):
    F = (np.roll(f, 1) + f) / 2
    return F


def CD4(f):
    F = (7 * (np.roll(f, 1) + f) - (np.roll(f, 2) + np.roll(f, -1))) / 12
    return F


def DOWN3(f):
    F = (2 * np.roll(f, 1) + 5 * f - np.roll(f, -1)) / 6
    return F


def P3_3(f):
    F = (2 * np.roll(f, 3) - 7 * np.roll(f, 2) + 11 * np.roll(f, 1)) / 6
    return F


def WENO1(f):
    # Precompute shifted arrays for smoothness indicators
    f_m2 = np.roll(f, 3)
    f_m1 = np.roll(f, 2)
    f_0 = np.roll(f, 1)
    f_p2 = np.roll(f, -1)

    # Calculate the smoothness indicators for the three stencils
    b1 = (13 / 12) * np.square(f_m2 - 2 * f_m1 + f_0) + np.square(f_m2 - 4 * f_m1 + 3 * f_0) / 4
    b2 = (13 / 12) * np.square(f_m1 - 2 * f_0 + f) + np.square(f_m1 - 4 * f_0 + 3 * f) / 4
    b3 = (13 / 12) * np.square(f_0 - 2 * f + f_p2) + np.square(f_0 - 4 * f + 3 * f_p2) / 4

    # Calculate the non-linear weights
    eps = 1e-6 # Makes sure denominator is never zero
    weights = np.array([1/10, 3/5, 3/10])
    smoothness = np.array([b1, b2, b3])
    alpha = weights / np.square(eps + smoothness)
    omega = alpha / np.sum(alpha, axis=0)

    F1 = calc_flux(f, 'P3_3-0')
    F2 = calc_flux(f, 'P3_2-1')
    F3 = calc_flux(f, 'P3_1-2')

    F = omega[0] * F1 + omega[1] * F2 + omega[2] * F3
    return F