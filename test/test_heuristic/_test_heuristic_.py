# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import pandas as pd
import numpy as np
from src.basenet.metaheuristic import BaseNetRandomSearch, BaseNetHeuristic, BaseNetGenetic, BaseNetPso
import hvplot.pandas
import holoviews as hv
# -----------------------------------------------------------


def my_callback(epoch_number, population, score):
    print(f'Epoch {epoch_number}; best: {max(score)}.')
    return True


def test_fitness(*args, **kwargs):
    the_ind = np.array(args)
    the_sum = np.sum(the_ind ** 2, axis=-1)
    return the_sum


def rule_first_is_maximum(conditionant: list, condition: list):
    the_condicionante = conditionant[0]
    for condicion in condition:
        if the_condicionante < condicion:
            return False
    return True


def rule_last_is_minimum(conditionant: list, condition: list):
    the_condicionante = conditionant[0]
    for condicion in condition:
        if the_condicionante > condicion:
            return False
    return True


def my_plot(pp: pd.DataFrame, selector_individual: int, selector_epoch: str):
    _pp_ = pp[pp.ts == selector_epoch].reset_index()
    _pp_ = _pp_.sort_values(by=['param0'])
    ifix = _pp_.hvplot.violin(y='fitness', by='param0', legend=True,
                              violin_color=hv.dim('param0').str(), cmap='rainbow', grid=True)
    return ifix


def problem(bnh):
    bnh.add_parameter(parameter_type='integer', maximum=20)
    [bnh.add_parameter(minimum=0, maximum=10) for _ in range(5)]
    bnh.add_plot(my_plot, name='PCA parameter 0')
    bnh.add_rule([0], [1, 2, 3, 4, 5], rule_first_is_maximum)
    # bnh.add_rule([5], [0, 1, 2, 3, 4], rule_last_is_minimum)
    population, score = bnh.fit(20, callback=my_callback, objective=900)
    print(f'Best individual: {population[0]}:{score[0]}')


def test_rs():
    bnh = BaseNetRandomSearch(test_fitness,
                              number_of_individuals=900,
                              new_individuals_per_epoch=300,
                              ray_ip='192.168.79.101',
                              runtime_env={'working_dir': '../'},
                              computational_cores=10)
    problem(bnh)


def test_genetic():
    bnh = BaseNetGenetic(test_fitness,
                         number_of_individuals=900,
                         new_individuals_per_epoch=300,
                         ray_ip='192.168.79.101',
                         runtime_env={'working_dir': '../'},
                         computational_cores=10,
                         mutation_variance=1,
                         )
    problem(bnh)


def pso_plot(pp: pd.DataFrame, selector_individual: int, selector_epoch: str):
    _pp_ = pp[pp.ts == selector_epoch].reset_index()
    ifix = _pp_.hvplot.scatter(x='param0', y='param1', by='fitness', cmap='rainbow', grid=True, xlim=(-5, 105),
                               ylim=(-55, 105))
    return ifix


def test_pso():
    pso = BaseNetPso(test_fitness,
                     number_of_individuals=10,
                     # ray_ip='192.168.79.101',
                     # runtime_env={'working_dir': '../'},
                     computational_cores=10,
                     inertia=2)
    pso.add_parameter(parameter_type='integer', maximum=100)
    pso.add_parameter(minimum=-50, maximum=100)
    pso.add_plot(pso_plot, name='PSO monitoring')
    population, score = pso.fit(20, objective=20_000)
    print(f'Best individual: {population[0]}: {score[0]}.')


if __name__ == '__main__':
    test_pso()
    # test_genetic()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
