import os
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import numba
import numpy as np


class FractalType(Enum):
    Julia = 1
    Mandelbrot = 2


class FractalGenerator:
    """ Creates a single fractal object and either returns it as as a numpy array, plot it or persists it as an pgn
            image. The output of this class is used by FractalTrainingValidationSet to generate training/val sets
            Args:
                None
        """

    def __init__(self):
        self.type_ = None
        self.fractal = None

#     @numba.jit(nopython=True)
    def create_julia(self, complex_function=lambda z: np.sin(z ** 4 + 1.41),
                     n=256, xlim=(-2, 2), ylim=(-2, 2), thr=2, max_iter=10):
        fractal = np.zeros((n, n), dtype='complex')
        x_space = np.linspace(xlim[0], xlim[1], n)
        y_space = np.linspace(ylim[0], ylim[1], n)
        for ix, x in enumerate(x_space):
            for iy, y in enumerate(y_space):
                for i in range(max_iter):
                    if i == 0: z = complex(x, y)
                    z = complex_function(z)
                    if np.abs(z) >= thr: z = thr; break
                fractal[ix, iy] = z
        self.fractal = np.abs(fractal)
        self.type_ = FractalType.Julia
        return self

#     @numba.jit(nopython=True)
    def create_mandelbrot(self, n=256, xlim=(-2, 0.5), ylim=(-1, 1), thr=2, max_iter=10):
        fractal = np.zeros((n, n), dtype='complex')
        x_space = np.linspace(xlim[0], xlim[1], n)
        y_space = np.linspace(ylim[0], ylim[1], n)
        for ix, x in enumerate(x_space):
            for iy, y in enumerate(y_space):
                for i in range(max_iter):
                    if i == 0: z = 0
                    z = z ** 2 + complex(x, y)
                    if np.abs(z) >= thr: z = thr; break
                fractal[ix, iy] = z
        self.fractal = np.abs(fractal.transpose())
        self.type_ = FractalType.Mandelbrot
        return self

    def plot(self, clim=None, **kwargs):
        if self.fractal is None:
            print('Nothing to plot. Generate a fractal first.')
            return None
        plt.matshow(self.fractal, **kwargs)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.clim(clim)
        return plt.gcf()

    def persist_plot(self, filename, container, clim=None, **kwargs):
        if not os.path.isdir(container): os.mkdir(container)
        self.plot(clim=clim, **kwargs)
        plt.savefig(str(Path(container) / filename), png='png', dpi=None)
        plt.close(plt.gcf())
