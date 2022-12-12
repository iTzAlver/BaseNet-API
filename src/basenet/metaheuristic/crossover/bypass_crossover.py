# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import tensorflow as tf


# -----------------------------------------------------------
def bypass_crossover(number_of_new_individuals, tf_population) -> tf.Tensor:
    """
    This crossover function returns the population. There is no crossover. It is an util function.
    :param number_of_new_individuals: Number of new individuals.
    :param tf_population: The current population.
    :return: The best number_of_individuals in the population.
    """
    return tf_population[number_of_new_individuals:]
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
