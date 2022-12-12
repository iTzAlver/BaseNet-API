# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import tensorflow as tf
import numpy as np
import itertools
import random


# -----------------------------------------------------------
def borderline_initializer(number_of_individuals: int, constraints) -> tf.Tensor:
    """
    This initializer creates an N dimensional mesh in the borderline of the parameters (minimum and maximum). Where N
    is the number of parameters.
    :param number_of_individuals: Number of individuals to be initialized.
    :param constraints: The problem constraints.
    :return: The initialized individuals.
    """
    if number_of_individuals > 1:
        order = int(np.ceil(np.exp(np.log(number_of_individuals) / len(constraints.parameters))))
    else:
        order = 1

    combinations = list(itertools.product(np.linspace(0, 1, order), repeat=len(constraints.parameters)))
    random.shuffle(combinations)
    true_combo = list()
    for combination in combinations:
        this_combo = list()
        for index, alpha in enumerate(combination):
            minimum = constraints.parameters[index].minimum
            maximum = constraints.parameters[index].maximum
            this_combo.append(minimum + alpha * (maximum - minimum))
        true_combo.append(this_combo)
        if len(true_combo) >= number_of_individuals:
            break

    return tf.convert_to_tensor(true_combo[:number_of_individuals], dtype=tf.float32)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
