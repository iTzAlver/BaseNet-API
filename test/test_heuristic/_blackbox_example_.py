# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import tensorflow as tf
import numpy as np
from src.basenet.metaheuristic import BaseNetRandomSearch


# -----------------------------------------------------------
def my_callback(epoch_number, population, score):
    print(f'Epoch {epoch_number}: {max(score)}.')
    return True


def blackbox_fitness(indi):
    the_ind = np.array(indi)
    the_sum_q = np.sum(the_ind ** 2, axis=-1)
    the_sum_m = np.sum(the_ind, axis=-1)
    return the_sum_m, the_sum_q


def myfitness_wrapper(*args, **kwargs):
    fitness_values = list()
    for individual in args:
        fit, dev = blackbox_fitness(individual)
        fitness_values.append(fit * 1 + dev * 1)
    return np.array(fitness_values)


def blackbox():
    bnh = BaseNetRandomSearch(myfitness_wrapper,
                              number_of_individuals=300,
                              new_individuals_per_epoch=100,
                              ray_ip='192.168.79.101',
                              runtime_env={'working_dir': '../'},
                              computational_cores=10)
    [bnh.add_parameter(parameter_type='real', minimum=-1000, maximum=1000) for _ in range(380)]
    bnh.fit(20, callback=my_callback)


if __name__ == '__main__':
    blackbox()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
