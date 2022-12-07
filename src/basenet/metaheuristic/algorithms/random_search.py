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
from ..mutation import basic_mutation
from ..initialization import random_initializer
from ..selection import elitist_selection
# -----------------------------------------------------------


class BaseNetRandomSearch(BaseNetHeuristic, ABC):
    """
    This class implements a RandomSearch as BaseNetHeuristic does. However, BaseNetHeuristic is designed to be an
    abstract class and BaseNetRandomSearch is built to be a user class. The random search algorithm tries random
    individuals with random mutation (there is no crossover between individuals).
    """
    @staticmethod
    def initializer(number_of_individuals: int, constraints: HeuristicConstraints) -> tf.Tensor:
        return random_initializer(number_of_individuals, constraints)

    @staticmethod
    def crossover(number_of_new_individuals: int, population: tf.Tensor) -> tf.Tensor:
        return basic_mutation(number_of_new_individuals, population)

    @staticmethod
    def selection(new_individuals: tf.Tensor, population: tf.Tensor) -> tf.Tensor:
        return elitist_selection(new_individuals, population)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
