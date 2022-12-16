# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
from abc import ABC
import tensorflow as tf
from ..basis import BaseNetHeuristic, HeuristicConstraints
from ..initialization import random_initializer
from .._utils.utils import get_algorithm_parameters
# -----------------------------------------------------------


class BaseNetCro(BaseNetHeuristic, ABC):
    def __init__(self, *args, **kwargs):
        """
        The BaseNetCro algorithm includes the CRO algorithm with the following configuration:
            initializer: random_initializer
            crossover: cro_crossover_included
            mutation: cro_crossover_included
            selection: cro_selection_included
        :param args: Must include the fitness function as first parameter.
        :param kwargs: Contains the genetic algorithm parameters:
        """
        cro_type, crossover_rate, mutation_variance, kwargs = \
            get_algorithm_parameters(cro_type='dynamic',
                                     crossover_rate=1.,
                                     mutation_variance=1.,
                                     kwargs=kwargs)
        self.cro = CroAlgorithm(cro_type=cro_type)
        super().__init__(*args, **kwargs)

    @staticmethod
    def initializer(number_of_individuals: int, constraints: HeuristicConstraints) -> tf.Tensor:
        return random_initializer(number_of_individuals, constraints)

    def crossover(self, number_of_new_individuals: int, population: tf.Tensor) -> tf.Tensor:
        return self.cro.cro_crossover(number_of_new_individuals, population)

    def selection(self, new_individuals: tf.Tensor, population: tf.Tensor) -> tf.Tensor:
        return self.cro.cro_selection(new_individuals, population)


class CroAlgorithm:
    def __init__(self, cro_type='dynamic'):
        self.cro_type = cro_type

    def cro_crossover(self, number_of_new_individuals, population):
        return population[:number_of_new_individuals]

    def cro_selection(self, new_individuals, population):
        return population
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
