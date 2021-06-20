import numba
import numpy as np

from fractalGAN.fractal_generator import FractalGenerator


class FractalTrainingValidationSet:
    """ Creates the training set necessary for the GAN algorithm by generating
        random samples from the same fractal family set  """

    def __init__(self, n_examples=30, n=256, directory='dataset'):
        self.n_examples = n_examples
        self.n = n
        self.directory = directory

#     @numba.jit(nopython=True, parallel=True)
    def generate_examples(self):
        for i in range(self.n_examples):
            frac = FractalGenerator()
            a, b = self.complex_function_parameter_sample()
            frac.create_julia(lambda z: self.complex_function(z, a=a, b=b), n=self.n)
            frac.persist_plot(filename='frac' + str(i), container=self.directory)

    @staticmethod
    def complex_function(z, a=0.0, b=0.0):
        """ function that defines the fractal"""
        f= np.sin(z** 4 + 1.41 + a)*np.exp((2.4+b)*1.J)
        return f

    @staticmethod
    def complex_function_parameter_sample():
        """ random parameter sample from the fractal 'family' set defined by the (a,b) ranges """
        return np.random.uniform(0.0, 2*np.pi), np.random.uniform(0.0, 0.1)
