# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import tensorflow as tf
import numpy as np


# -----------------------------------------------------------
def gaussian_initializer(number_of_individuals: int, constraints, interest_points, variance: float) -> tf.Tensor:
    """
    This initializer creates a random population around the selected interest points.
    :param number_of_individuals: The number of individuals in the population.
    :param constraints: The problem constraints.
    :param interest_points: The interest points where the individuals will be initialized around.
    :param variance: The variance towards the interest points.
    :return: The initial population.
    """
    indis = list()
    for individual in range(number_of_individuals):
        indi = list()
        interest_point = interest_points[np.random.randint(0, len(interest_points))]
        for parameter, interest in zip(constraints.parameters, interest_point):
            value = interest + np.sqrt(variance) * np.random.randn()
            if parameter.type == 'integer' or parameter.type == 'categorical':
                value = np.round(value)
            indi.append(value)
        indis.append(indi)
    return tf.convert_to_tensor(indis, dtype=tf.float32)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
