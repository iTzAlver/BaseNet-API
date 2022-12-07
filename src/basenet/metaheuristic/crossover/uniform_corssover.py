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
def uniform_crossover(number_of_new_individuals, tf_population, mask=None) -> tf.Tensor:
    new_individuals = list()
    population = tf_population.numpy()
    for _ in range(round(number_of_new_individuals / 2)):
        parents = population[np.random.randint(0, len(population) - number_of_new_individuals, 2)]
        offspring_0 = list()
        offspring_1 = list()
        if mask is not None:
            selector_0 = mask
        else:
            selector_0 = np.random.randint(0, 2, len(parents[0]))

        for n_param, selection0 in enumerate(selector_0):
            offspring_0.append(parents[selection0, n_param])
            offspring_1.append(parents[1 if selection0 == 0 else 0, n_param])

        new_individuals.append(offspring_0)
        new_individuals.append(offspring_1)
    return tf.convert_to_tensor(new_individuals, dtype=tf.float32)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
