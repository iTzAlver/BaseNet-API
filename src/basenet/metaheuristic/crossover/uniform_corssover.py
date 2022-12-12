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
def uniform_crossover(number_of_new_individuals, tf_population, mask=None, crossover_rate: float = 0.5) -> tf.Tensor:
    """
    This function implements the uniform crossover. We generate a mask (or provided) with a uniform distribution.

    Let's assume we have a crossover_rate of 0.5, where the mask elements flip a coin:

    Parents and mask:
    mask = [0, 1, 1, 1, 0, 0, 1, 0] (Random)
    p1 = [A, B, C, D, E, F, G, H]
    p2 = [I, J, K, L, M, N, O, P]

    Generates:
    o1 = [A, J, K, L, E, F, O, H]
    o2 = [I, B, C, D, M, N, G, P]

    :param number_of_new_individuals: Number of new individuals in the population.
    :param tf_population: The current population.
    :param mask: A crossover mask if given.
    :param crossover_rate: The probability that a parameter is chosen for crossover for random mask.
    :return: The new individuals as a result of the crossover.
    """
    new_individuals = list()
    population = tf_population.numpy()
    for _ in range(round(number_of_new_individuals / 2)):
        parents = population[np.random.randint(0, len(population) - number_of_new_individuals, 2)]
        offspring_0 = list()
        offspring_1 = list()
        if mask is not None:
            selector_0 = mask
        else:
            selector_0 = np.random.choice(2, len(parents[0]), p=[1 - crossover_rate, crossover_rate])

        for n_param, selection0 in enumerate(selector_0):
            offspring_0.append(parents[selection0, n_param])
            offspring_1.append(parents[1 if selection0 == 0 else 0, n_param])

        new_individuals.append(offspring_0)
        new_individuals.append(offspring_1)
    return tf.convert_to_tensor(new_individuals, dtype=tf.float32)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
