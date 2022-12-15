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
from ..selection import elitist_selection
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
        self.mask, self.crossover_rate, self.mutation_variance = get_algorithm_parameters(kwargs=kwargs)
        super().__init__(*args, **kwargs)

    @staticmethod
    def initializer(number_of_individuals: int, constraints: HeuristicConstraints) -> tf.Tensor:
        return random_initializer(number_of_individuals, constraints)

    def crossover(self, number_of_new_individuals: int, population: tf.Tensor) -> tf.Tensor:
        return cro_crossover(number_of_new_individuals, population)

    @staticmethod
    def selection(new_individuals: tf.Tensor, population: tf.Tensor) -> tf.Tensor:
        return elitist_selection(new_individuals, population)


def cro_crossover(number_of_new_individuals, population):

    return population[:number_of_new_individuals]


def cro_selection(new_individuals, population):

    return population
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
