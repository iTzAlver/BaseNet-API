# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
from abc import ABC
import tensorflow as tf
import numpy as np
from ..basis import BaseNetHeuristic, HeuristicConstraints
from ..initialization import random_initializer
from ..selection import no_selection
from .._utils.utils import get_algorithm_parameters


# -----------------------------------------------------------


class BaseNetPso(BaseNetHeuristic, ABC):
    def __init__(self, *args, **kwargs):
        """
        The BaseNetPso algorithm includes the PSO algorithm with the following configuration:
            initializer: random_initializer
            crossover: PSO has no crossover, it is only mutation by differences.
            mutation: pso_crossover_included
            selection: PSO has no selection, it is only mutation by differences.
        :param args: Must include the fitness function as first parameter.
        :param kwargs: Contains the genetic algorithm parameters:
        :param inertia: The PSO inertia:
            Velocity (v) of individual i in the population (pop) in parameter (p) in epoch (t):
                v[i][p][t + 1] =    inertia * v[i][p][t] +
                                    social_factor * (pop[i=best][p][t] - pop[i][p][t]) +
                                    cognition_factor * (pop[i][p][t=best] - pop[i][p][t])
        """
        inertia, cognition, social, kwargs = get_algorithm_parameters(inertia=.5,
                                                                      cognition_factor=1.,
                                                                      social_factor=1.,
                                                                      kwargs=kwargs)
        kwargs['new_individuals_per_epoch'] = kwargs['number_of_individuals']
        self.pso = PSODifferentials(inertia, cognition, social)
        super().__init__(*args, **kwargs)

    @staticmethod
    def initializer(number_of_individuals: int, constraints: HeuristicConstraints) -> tf.Tensor:
        return random_initializer(number_of_individuals, constraints)

    def crossover(self, number_of_new_individuals: int, population: tf.Tensor) -> tf.Tensor:
        return self.pso.pso_mutation(population, self.score, self.identification)

    @staticmethod
    def selection(new_individuals: tf.Tensor, population: tf.Tensor) -> tf.Tensor:
        return no_selection(new_individuals, population, which='new')


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        AID CLASSES                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #

class PSODifferentials:
    def __init__(self, inertia: float = 2., cognition: float = 1., social: float = 1.):
        self.inertia: float = inertia
        self.cognition: float = cognition
        self.social: float = social

        self.mapping: (list[tuple[tf.Tensor, float, tf.Tensor]], None) = None

    def pso_mutation(self, population: tf.Tensor, score: tf.Tensor, ids: tf.Tensor):

        if self.mapping is None:
            self.mapping = [()] * len(population)
            for individual, fitness, name in zip(population, score, ids):
                # noinspection PyTypeChecker
                self.mapping[int(name)] = (individual, fitness, tf.zeros(population.shape[1]))
                # best_solution[1 x np], best_score[1 x 1], current_speed[1 x np]

        speeds, bs = self.remap(ids)
        speed = self.compute_speed(population, speeds, bs)
        self.update_map(population, ids, speed, score)
        new_population = self.apply_speed(speed, population)

        return tf.convert_to_tensor(new_population, dtype=tf.float32)

    def compute_speed(self, population: tf.Tensor, speeds: tf.Tensor, best_sols: tf.Tensor) -> tf.Tensor:
        r0, r1 = np.random.random(2)
        c0 = r0 * self.cognition
        c1 = r1 * self.social
        w = self.inertia
        cognitive: tf.Tensor = tf.multiply(c0, (best_sols - population))
        social: tf.Tensor = tf.multiply(c1, (population[0] - population))
        inertia: tf.Tensor = tf.multiply(w, speeds)
        speed: tf.Tensor = inertia + cognitive + social
        return speed

    @staticmethod
    def apply_speed(speed: tf.Tensor, population: tf.Tensor):
        """
        x[t + 1] = x[t] + v[t + 1]
        """
        return population + speed

    def remap(self, ids: tf.Tensor):
        speeds = list()
        best_positions = list()

        for name in ids:
            this_best_position, _, this_speed = self.mapping[int(name)]
            speeds.append(this_speed)
            best_positions.append(this_best_position)

        _retval = (tf.convert_to_tensor(speeds, dtype=tf.float32),
                   tf.convert_to_tensor(best_positions, dtype=tf.float32))
        return _retval

    def update_map(self, population, ids, speeds, scores):
        for individual, name, speed, score in zip(population, ids, speeds, scores):
            this_best_position, this_score, this_speed = self.mapping[int(name)]
            if score > this_score:
                self.mapping[int(name)] = (individual, score, speed)
            else:
                self.mapping[int(name)] = (this_best_position, this_score, speed)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
