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
def stochastic_mutation(population: tf.Tensor, mutation_rate: float, mutation_variance: float,
                        distribution=np.random.randn, distribution_args: tuple = ()) -> tf.Tensor:
    new_indis = list()
    for individual in population.numpy():
        new_indi = list()
        for parameter in individual:
            if np.random.choice(2, p=[1 - mutation_rate, mutation_rate]):
                noise = np.sqrt(mutation_variance) * distribution(*distribution_args)
            else:
                noise = 0
            new_indi.append(parameter + noise)
    return tf.convert_to_tensor(new_indis, dtype=tf.float32)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
