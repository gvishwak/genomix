""" A Python implementation of real-valued genetic algorithms for hyperparameter optimization in machine learning, particularly focused on applications in chemistry and materials science.
"""

__name__ = "Genomix"
__version__ = "1.0"
__author__ = "Gaurav Vishwakarma"


__all__ = [
    'GeneticAlgorithm',
]


from .genetic_algorithm import GeneticAlgorithm
from .__version__ import __version__


