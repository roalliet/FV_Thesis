import numpy as np
from enum import Enum


class FunctionType(Enum):
    Block = 0
    Sin = 1


def init(p):
    # Determine dx and initialize cell coordinates and cell averages
    x = np.arange(p.grid_size) * p.dx + p.xmin + p.dx / 2 
    f = np.zeros(p.grid_size)

    # Determine cell averages based on analytical function
    if p.func.value == 0:
    # Block function
        f[p.grid_size //5:p.grid_size //5 * 3] = 1 
    if p.func.value == 1:
    # Sine wave
    # Face averaged values are f = sin(x_{i+1/2})
        f = np.sin(x * 2 * np.pi) 
    return f, x


def exact(p, grid_size=0):
    # Determine whether a separate gridsize was defined for the exact solution
    if grid_size == 0:
        grid_size = p.grid_size
    # Determine dx and initialize cell coordinates
    dx = (p.xmax-p.xmin)/(grid_size)
    x = np.arange(grid_size) * dx + p.xmin + dx / 2 #required for plotting function later on

    # Determine cell averages based on analytical function at time t
    if p.func.value == 0:
        # Block function
        f = np.array([1 if ((x[i] + p.v * p.duration + 1) % 2 - 1 < 0.2) & ((x[i] + p.v * p.duration + 1) % 2 - 1 > -0.6) else 0 for i in range(grid_size)])
    elif p.func.value == 1:
        # Sine wave
        f = np.sin((x- p.duration * p.v) * 2 * np.pi)

    return f, x