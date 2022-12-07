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
def binary_mutation(population: tf.Tensor, mutation_rate: float) -> tf.Tensor:
    new_indis = list()
    lp = len(population[0])
    for individual in population.numpy():
        noise = 0.5 * np.random.randn(lp)
        new_indis.append(individual + noise)
    return tf.convert_to_tensor(new_indis, dtype=tf.float32)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
