from enum import Enum


class Parameters():
    def __init__(self, grid_size: int, xmin: float, xmax: float, duration: float, v: float, C_max: float, evotype: str, stencil: str, func: Enum, mult_stencil: list):
        self.grid_size = grid_size
        self.xmin = xmin
        self.xmax = xmax
        self.dx = (xmax - xmin) / grid_size
        self.duration = duration
        self.t = 0
        self.v = v
        self.C_max = C_max
        self.evotype = evotype
        self.stencil = stencil
        self.func = func
        self.dt = self.C_max * self.dx / self.v
        self.mult_stencil = mult_stencil
        self.mult_len = len(mult_stencil)

    def new_grid(self, grid_size):
        self.grid_size = grid_size
        self.dx = (self.xmax - self.xmin) / self.grid_size
        self.dt = self.C_max * self.dx / self.v
        self.t = 0

    def new_stencil(self, ms):
        self.t = 0
        self.dt = self.C_max * self.dx / self.v
        self.stencil = ms