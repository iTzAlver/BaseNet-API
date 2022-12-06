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
def basic_mutation(number_of_new_individuals: int, population: tf.Tensor) -> tf.Tensor:
    mutatives = np.random.randint(0, len(population), number_of_new_individuals)
    new_indis = list()
    for mutative in mutatives:
        individual = population[mutative].numpy()
        for npp, parameter in enumerate(individual):
            individual[npp] = parameter * (1 + np.random.randn() * 0.5)
        new_indis.append(individual)
    return tf.convert_to_tensor(new_indis)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
