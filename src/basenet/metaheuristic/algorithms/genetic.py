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
from ..crossover import multipoint_crossover, uniform_crossover, arithmetic_crossover
from ..initialization import random_initializer, borderline_initializer, gaussian_initializer
from ..selection import elitist_selection
from ..mutation import gaussian_mutation, binary_mutation
# -----------------------------------------------------------


class BaseNetGenetic(BaseNetHeuristic, ABC):
    """
    This class implements a RandomSearch as BaseNetHeuristic does. However, BaseNetHeuristic is designed to be an
    abstract class and BaseNetRandomSearch is built to be a user class. The random search algorithm tries random
    individuals with random mutation (there is no crossover between individuals).
    """
    def __init__(self, *args, **kwargs):

        # For multipoints.
        if 'multipoints' in kwargs:
            nmp = kwargs.pop('multipoints')
            if nmp > 0:
                self.multipoint_points = nmp
            else:
                self.multipoint_points = 1
        else:
            self.multipoint_points = 1

        # For gaussian mutation.
        if 'mutation_var' in kwargs:
            nmp = kwargs.pop('mutation_var')
            if nmp > 0:
                self.mutation_variance = nmp
            else:
                self.mutation_variance = 0
        else:
            self.mutation_variance = 0

        # For binary mutation.
        if 'mutation_rate' in kwargs:
            nmp = kwargs.pop('mutation_rate')
            if nmp >= 1:
                self.mutation_rate = round(nmp)
            else:
                self.mutation_rate = 0
        else:
            self.mutation_rate = 0
        super().__init__(*args, **kwargs)

    @staticmethod
    def initializer(number_of_individuals: int, constraints: HeuristicConstraints) -> tf.Tensor:
        return random_initializer(number_of_individuals, constraints)

    def crossover(self, number_of_new_individuals: int, population: tf.Tensor) -> tf.Tensor:
        non_muted = uniform_crossover(number_of_new_individuals, population)
        muted = gaussian_mutation(non_muted, self.mutation_variance)
        return muted

    @staticmethod
    def selection(new_individuals: tf.Tensor, population: tf.Tensor) -> tf.Tensor:
        return elitist_selection(new_individuals, population)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
