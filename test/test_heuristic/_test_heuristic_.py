# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import pandas as pd
import numpy as np
from src.basenet.metaheuristic import BaseNetHeuristic
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


def my_plot(pp: pd.DataFrame, selector_individual: int, selector_epoch: str):
    # _pp = pp[pp.individual == interactive[0]].reset_index()
    _pp_ = pp[pp.ts == selector_epoch].reset_index()
    # ifix = _pp_.hvplot.scatter(x='param0', y='fitness', grid=True)
    _pp_ = _pp_.sort_values(by=['param0'])
    ifix = _pp_.hvplot.violin(y='fitness', by='param0', legend=True,
                              violin_color=hv.dim('param0').str(), cmap='rainbow', grid=True)
    return ifix


def main():
    bnh = BaseNetHeuristic(test_fitness,
                           number_of_individuals=900,
                           new_individuals_per_epoch=300,
                           ray_ip='192.168.79.101',
                           runtime_env={'working_dir': '../'},
                           computational_cores=6)
    bnh.add_parameter(parameter_type='integer', maximum=20)
    [bnh.add_parameter(minimum=0, maximum=10) for _ in range(5)]
    bnh.add_plot(my_plot, name='PCA parameter 0')
    # bnh.add_rule([0], [1, 2, 3, 4, 5], rule_first_is_maximum)
    population, score = bnh.fit(20, callback=my_callback)
    print(f'Best individual: {population[0]}:{score[0]}')


if __name__ == '__main__':
    main()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
