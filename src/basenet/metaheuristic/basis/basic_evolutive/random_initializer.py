# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import numpy as np
import tensorflow as tf


# -----------------------------------------------------------
def random_initializer(number_of_individuals: int, constraints) -> tf.Tensor:
    indis = list()
    for individual in range(number_of_individuals):
        indi = list()
        for parameter in constraints.parameters:
            if parameter.type == 'real':
                value = np.random.uniform(parameter.minimum, parameter.maximum)
            elif parameter.type == 'integer' or parameter.type == 'categorical':
                value = np.random.randint(parameter.minimum, parameter.maximum)
            else:
                value = 0.0
            indi.append(value)
        indis.append(indi)
    return tf.convert_to_tensor(indis)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
