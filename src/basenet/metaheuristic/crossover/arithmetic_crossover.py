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
def arithmetic_crossover(number_of_new_individuals, tf_population,
                         crossover_rate: float = 0.2, alpha: float = 0.2) -> tf.Tensor:
    """
    This function implements the arithmetic crossover:

    p1 = [A, B, C]
    p2 = [D, E, F]

    o1 = [alpha * A + (1-alpha) * D, alpha * B + (1-alpha) * E, alpha * C + (1-alpha) * F]
    o2 = [alpha * D + (1-alpha) * A, alpha * E + (1-alpha) * B, alpha * F + (1-alpha) * C]

    :param number_of_new_individuals: Number of new individuals to be created.
    :param tf_population: Input population.
    :param crossover_rate: Probability that a parameter in the individual mutates.
    :param alpha: Fidelity to the parent (between 0 and 1).
    :return: The new individuals as a result of the crossover.
    """
    new_individuals = list()
    population = tf_population.numpy()
    for _ in range(round(number_of_new_individuals / 2)):
        parents = population[np.random.randint(0, len(population) - number_of_new_individuals, 2)]
        offspring_0 = list()
        offspring_1 = list()
        for parameter_0, parameter_1 in zip(*parents):
            if np.random.choice(2, p=[crossover_rate, 1 - crossover_rate]):
                offspring_0.append(alpha * parameter_0 + (1 - alpha) * parameter_1)
                offspring_1.append(alpha * parameter_1 + (1 - alpha) * parameter_0)
            else:
                offspring_0.append(parameter_0)
                offspring_1.append(parameter_1)
        new_individuals.append(offspring_0)
        new_individuals.append(offspring_1)
    return tf.convert_to_tensor(new_individuals[:number_of_new_individuals], dtype=tf.float32)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
