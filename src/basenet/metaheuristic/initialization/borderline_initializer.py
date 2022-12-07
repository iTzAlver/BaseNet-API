# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import tensorflow as tf
import numpy as np
import itertools
# from src.basenet.metaheuristic.basis.basic_evolutive import random_initializer


# -----------------------------------------------------------
def borderline_initializer(number_of_individuals: int, constraints) -> tf.Tensor:

    indis = list()
    binaries = list(itertools.product([0, 1], repeat=len(constraints.parameters)))
    rusher = 1

    while len(indis) < number_of_individuals:
        localls = list()
        for parameter in constraints.parameters:
            localls.append((parameter.minimum / rusher, parameter.maximum / rusher))
        localls = np.array(localls)

        for binary in binaries:
            indi = list()
            for bit, locall in zip(binary, localls):
                indi.append(locall[bit])
            indis.append(indi)
        rusher += 1

    return tf.convert_to_tensor(indis[:number_of_individuals])
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
