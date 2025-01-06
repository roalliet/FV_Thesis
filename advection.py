import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.stats import linregress
from enum import Enum


class FunctionType(Enum):
    Polynomial = 0
    Sin = 1
    Cos = 2
    Flat = 3
    Block = 4
    Kink = 5
    Tanh = 6
    RapSin = 7
    

def init():
    X_f = np.arange(grid_size+1) * dx + xmin
    X_f = X_f  # Apply scaling and shifting to X_f
    X_v =  X_f + dx / 2
    return X_f, X_v


def ana_sol(X_f, v, t, func : Enum):
    # Polynomial
    if func.value == 0: 
        X_f2 = X_f * X_f
        X_f3 = X_f2 * X_f
        X_f4 = X_f3 * X_f
        X_f5 = X_f4 * X_f
        X_f6 = X_f5 * X_f
        F_int = X_f6/6 - X_f5 + 5/4 * X_f4 + 5/3 * X_f3 - 3 * X_f2 + 4*X_f
        F_v = (F_int[1:] - F_int[:-1])/dx
        F_f = X_f5 - 5 * X_f4 + 5 * X_f3 + 5 * X_f2 - 6 * X_f + 4 

    # Sine
    elif func.value == 1:
        cos = np.cos(X_f) 
        cos3 = np.cos(X_f) * np.cos(X_f)* np.cos(X_f)
        F_f = np.sin(X_f) * np.sin(X_f) * np.sin(X_f)
        F_v = ((cos3[1:]-cos3[:-1])/3 - ((cos[1:]-cos[:-1])))/dx

    # Cosine
    elif func.value == 2:
        F_f = np.cos(X_f)
        sin = np.sin(X_f)
        F_v = (sin[1:]-sin[:-1])/dx

    # Flat
    elif func.value == 3:
        F_f = np.ones(grid_size+1)
        F_v = (X_f[1:]-X_f[:-1])/dx
    
    # Block function
    elif func.value ==4:
        F_f = np.zeros_like(X_f)
        F_f[grid_size //10 * 4:grid_size //10 * 7] = 1 
        F_v = np.zeros(len(X_f)-1)
        for i in range(len(F_v)):
            F_v[i] = ((1-w) * F_f[i+1] + w * F_f[i])

    # Kink
    elif func.value == 5:
        # Initiate lists
        F_f = np.zeros_like(X_f)

        # Determine antiderivatives
        cosh = np.log(np.cosh(100*X_f))/100
        cubic = 8 / 3 * X_f * X_f * X_f - 12 * X_f * X_f + 17 * X_f

        # Tanh for x<1
        fval = np.where(X_f >= 1)[0][0]
        F_f[:fval] = np.tanh(100*X_f[:fval])
        F_v1 = (cosh[1:fval] - cosh[:fval-1])/dx

        # Parabola for x>1
        F_f[fval:] = 8 * X_f[fval:] * X_f[fval:] - 24 * X_f[fval:] + 17
        F_v3 = (cubic[fval + 1:] - cubic[fval:-1]) / dx 

        # Transition value at x = 1
        F_v2 = (cubic[fval] - cosh[fval - 1]) / dx

        F_v = np.concatenate([F_v1, np.array([F_v2]), F_v3])
    elif func.value == 6:
        # Determine antiderivatives
        cosh = np.log(np.cosh(100 * (X_f - 1))) / 100

        F_f = np.tanh(100 * (X_f - 1))
        F_v = (cosh[1:] - cosh[:-1])/dx
    elif func.value == 7:
        X_f += v *t
        factor = 1
        F_f = np.sin(X_f * factor)
        cos = -np.cos(X_f * factor) / factor
        F_v = (np.roll(cos, -1)-cos) / dx

    else:
        raise ValueError('Function does not exist, pick a different value')

    return F_f, F_v


def time_step():
    #determines the maximal value of dt based on the CFL criterion
    C_max = 0.3
    return C_max * dx / v


def UP4(F):
    #calculate the face integrated values of f
    F_f = (3 * np.roll(F, -1) + 13 * F - 5 * np.roll(F, 1) + np.roll(F, 2)) / 12
    return F_f


def advance_time(F_v, t, var):
    #determine the time step dt
    dt = time_step()

    #ensures code always stops exactly at specified end time "duration"
    if t+dt > duration:
        dt = duration-t

    #determine the face averaged values
    F_f = var(F_v)

    #determine the volume integrated values after a half timestep
    F_half = F_v - dt / 2 * (F_f - np.roll(F_f, 1))
    
    F_f = var(F_half)

    F_v = F_v - dt * (F_f - np.roll(F_f, 1))

    return F_v, dt


def run(var, save=False):
    #create initial values 
    t = 0
    X_f, X_v = init()
    F_f, F_v = ana_sol(X_f, v, t, func)

    #evolve the distribution in time while updating the time
    while t < duration:
        F_v, dt = advance_time(F_v, t, var)
        t += dt
    
    #create analytical solution at end time
    F_f_ana, _ = ana_sol(X_f, v, t, func)

    #determine interpolation at final time
    F_f = var(F_v)

    # Create the plot
    fig, ax = plt.subplots()
    
    # Plot the initial function
    ax.plot(X_f, F_f, lw=2, label="numerical", color='red')
    ax.plot(X_f, F_f_ana, lw=2, label="analytical", color='blue', ls='-')
    ax.grid()
    ax.legend()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    # Set up spines
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()

    if save:
        plt.savefig(f'interpolation_plots/{func}_interpolation_using_{var.__name__}_{grid_size}.png', dpi=300, bbox_inches='tight')
    plt.show()


#initialise some parameters of the system
grid_size = 10000
v = 1
duration = 0
xmin = -np.pi
xmax = np.pi
ymin = 1.2
ymax = -1.2
dx = (xmax - xmin) / grid_size
func = FunctionType(7)
w = 1

run(UP4)
