# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
from basenet.metaheuristic import HeuristicOptimization, HeuristicConstraints
# -----------------------------------------------------------


def my_callback(epoch, pop, score):
    print(f'Epoch {epoch} finished. Best: {max(score)}.')
    if epoch > 30:
        return True
    return False


def main():
    const = HeuristicConstraints()
    const.add_parameter(parameter_shape=(), parameter_type=float, min_value=0, max_value=1)
    const.add_parameter(parameter_shape=(), parameter_type=int, min_value=0, max_value=1)
    const.add_parameter(parameter_shape=(), parameter_type=float, min_value=0, max_value=1)
    const.add_parameter(parameter_shape=(), parameter_type=float, min_value=0, max_value=1)
    const.add_rule((0, 1, 2), 3, operand=lambda x, y: sum(x) / len(x) > y)

    opt = HeuristicOptimization(10, 2, const, callback=my_callback)
    opt.fit(epoch=1000)
    best = opt.population[0]
    print(f'End of test. Best: {best}.')


if __name__ == '__main__':
    main()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
