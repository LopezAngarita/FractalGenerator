import numpy as np

from fractal_generator.fractal_generator import FractalGenerator


class FractalTrainingSet:
    """ Creates the training set necessary for the GAN algorithm by generating
        random samples from the same fractal family set

        Args:
        n_examples -- number of fractals to output
        n -- fractal will be of size n*n
        directory -- output directory
    """

    def __init__(self, n_examples=30, n=256, directory='dataset'):
        self.n_examples = n_examples
        self.n = n
        self.directory = directory

    def generate_examples(self):
        for i in range(self.n_examples):
            frac = FractalGenerator(n=self.n)
            a, b = self.complex_function_parameter_sample()
            frac.create_julia(lambda z: self.complex_function(z, a=a, b=b))
            frac.persist_plot(filename='frac' + str(i), container=self.directory)

    @staticmethod
    def complex_function(z, a=0.0, b=0.0):
        """ function that defines the fractal"""
        f = np.sin(z ** 4 + 1.41 + a) * np.exp((2.4 + b) * 1.J)
        return f

    @staticmethod
    def complex_function_parameter_sample():
        """ random parameter sample from the fractal 'family' set defined by the (a,b) ranges """
        return np.random.uniform(0.0, 2 * np.pi), np.random.uniform(0.0, 0.1)
