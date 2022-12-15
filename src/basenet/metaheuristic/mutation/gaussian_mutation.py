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
def gaussian_mutation(population: tf.Tensor, mutation_variance: float) -> tf.Tensor:
    """
    This function generates a gaussian mutation.
    :param population: The current individuals to be mutated.
    :param mutation_variance: The variance of the gaussian.
    :return: The mutated individuals.
    """
    noise: np.ndarray = np.sqrt(mutation_variance) * np.random.randn(*population.shape)
    new_indis = population + noise
    return tf.convert_to_tensor(new_indis, dtype=tf.float32)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
