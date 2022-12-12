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
    """
    This function implements the multipoint crossover:
    We select a number_of_points points in the individual.

    Let's assume it was 1 point in the middle:
    Parents:
    p1 = [A, B, C. D, E]
    p2 = [F, G, H. I, J]

    Generate:
    o1 = [A, B, C. I, J]
    o2 = [F, G, H. D, E]

    Let's assume it was 2 points in the middle:
    Parents:
    p1 = [A, B. C. D, E]
    p2 = [F, G. H. I, J]

    Generate:
    o1 = [A, B. H. D, E]
    o2 = [F, G. C. I, J]

    :param number_of_new_individuals: Number of individuals in the new population.
    :param tf_population: The current population.
    :param number_of_points: Number of points in the multipoint crossover method.
    :return: The new individuals as a result of the crossover.
    """
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
