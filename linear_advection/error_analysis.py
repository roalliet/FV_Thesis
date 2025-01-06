import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm


def calc_L1(f1, f2=None):
    #determine the L1-norm
    if f2 is None:
        L1 = norm(f1, 1)
    else:
        L1 = norm(f1-f2, 1)
    return L1


def calc_L2(f1, f2=None):
    #determine the L2-norm
    if f2 is None:
        L2 = norm(f1, 2)
    else:
        L2 = norm(f1-f2, 2)
    return L2


def calc_L_inf(f1, f2=None):
    #determine the L_infty-norm
    if f2 is None:
        L_inf = norm(f1, np.inf)
    else:
        L_inf = norm(f1-f2, np.inf)
    return L_inf


def calc_rel_L1(f, f_ex):
    return np.abs(calc_L1(f, f_ex)) / calc_L1(f_ex)


def calc_rel_L2(f, f_ex):
    return np.abs(calc_L2(f, f_ex)) / calc_L2(f_ex)


def calc_rel_L_inf(f, f_ex):
    return np.abs(calc_L_inf(f, f_ex)) / calc_L_inf(f_ex)