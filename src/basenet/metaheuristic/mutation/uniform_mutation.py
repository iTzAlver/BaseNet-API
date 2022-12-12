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
def uniform_mutation(population: tf.Tensor, mutation_rate: float, constraints) -> tf.Tensor:
    new_indis = list()
    for individual in population.numpy():
        new_indi = list()
        for parameter, cp in zip(individual, constraints.parameters):
            if np.random.choice(2, p=[1 - mutation_rate, mutation_rate]):
                new_indi.append(np.random.uniform(cp.minimum, cp.maximum))
            else:
                new_indi.append(parameter)
    return tf.convert_to_tensor(new_indis, dtype=tf.float32)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
