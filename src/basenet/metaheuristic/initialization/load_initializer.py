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
def load_initializer(number_of_individuals: int, constraints, path: str) -> tf.Tensor:
    """
    The load initializer takes a path as input as a np.array of individuals.
    :param number_of_individuals: The number of individuals in the population.
    :param constraints: The problem constraints.
    :param path: The path where the np.array of individuals is stored.
    :return: The initial population.
    """
    load_individuals = np.array(np.load(path))
    if not isinstance(load_individuals, np.ndarray):
        raise ValueError(f'BaseNetHeuristic:load_initializer: Cannot import {path} in the load_initializer. '
                         f'The input path does not contain a numpy array. It contains other type of object: '
                         f'{type(load_individuals)}')
    if len(load_individuals.shape) != 2:
        raise ValueError(f'BaseNetHeuristic:load_initializer: Cannot import {path} in the load_initializer. '
                         f'The input path does not contain a numpy array with 2 dimensions (population, parameters). '
                         f'It contains a numpy array but its shape is: {load_individuals.shape}.')
    elif load_individuals.shape[1] != len(constraints.parameters):
        raise ValueError(f'BaseNetHeuristic:load_initializer: Cannot import {path} in the load_initializer. '
                         f'The number of parameters in the input individuals do not match the'
                         f' constraints ({load_individuals.shape[1]} vs {len(constraints.parameters)}).')
    return tf.convert_to_tensor(load_individuals[:number_of_individuals], dtype=tf.float32)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
