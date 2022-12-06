# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import numpy as np
import tensorflow as tf
PARAMETER_TYPES = ['real', 'integer', 'categorical']


# -----------------------------------------------------------
class HeuristicConstraints:
    def __init__(self):
        self.parameters: list = list()
        self.rules: list = list()

    def add_parameter(self, parameter_type: str = 'real', minimum: any = 0, maximum: any = 0):
        self.parameters.append(HeuristicParameter(parameter_type, minimum, maximum))
        return self

    def add_rule(self, conditioned: list[int], conditions: list[int], operator):
        self.rules.append(HeuristicRule(conditioned, conditions, operator))
        return self

    def check_constraints(self, population: tf.Tensor):
        np_pop = population.numpy()
        respectless = set()
        for rule in self.rules:
            non_ruled = rule.check(np_pop)
            for non_rule in non_ruled:
                respectless.add(non_rule)
        return list(respectless)

    def apply_bindings(self, population: tf.Tensor):
        _population = tf.transpose(population)
        racks = list()
        for parameter, idea in zip(_population, self.parameters):
            rack = tf.clip_by_value(parameter, clip_value_min=idea.minimum, clip_value_max=idea.maximum)
            if idea.type == 'integer' or idea.type == 'categorical':
                rack = tf.round(rack)
            racks.append(rack)
        return tf.transpose(tf.convert_to_tensor(racks))


class HeuristicParameter:
    def __init__(self, parameter_type: str = 'real', minimum: any = 0, maximum: any = 0):
        self.type = parameter_type
        if parameter_type == 'real':
            self.minimum = float(minimum)
            self.maximum = float(maximum)
        if parameter_type == 'integer':
            self.minimum = int(minimum)
            self.maximum = int(maximum)
        elif parameter_type == 'categorical':
            self.minimum = 0
            self.maximum = int(maximum)


class HeuristicRule:
    def __init__(self, conditioned: list[int], conditions: list[int], operator):
        self.conditioned = conditioned
        self.conditions = conditions
        self.operator = operator

    def check(self, population: np.array):
        respectless = list()
        for respectless_index, individual in enumerate(population):
            conditioned = list(individual[self.conditioned])
            conditions = list(individual[self.conditions])
            if not self.operator(conditioned, conditions):
                respectless.append(respectless_index)
        return respectless
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
