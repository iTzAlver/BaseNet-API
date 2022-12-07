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
def multipoint_crossover(number_of_new_individuals, tf_population, number_of_points) -> tf.Tensor:
    new_individuals = list()
    population = tf_population.numpy()
    for _ in range(round(number_of_new_individuals / 2)):
        parents = population[np.random.randint(0, len(population) - number_of_new_individuals, 2)]
        multi_points = np.random.randint(0, len(parents[0]), number_of_points)
        toggle = 0
        offspring_0 = list()
        offspring_1 = list()
        for n_param, __ in enumerate(parents[0]):
            offspring_0.append(parents[toggle, n_param])
            offspring_1.append(parents[1 - toggle, n_param])
            if n_param in multi_points:
                toggle = 1 - toggle

        new_individuals.append(offspring_0)
        new_individuals.append(offspring_1)
    return tf.convert_to_tensor(new_individuals)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
