# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import tensorflow as tf
from tensorflow.python.client import device_lib
from ..deeplearning import BaseNetModel


# -----------------------------------------------------------
# TODO
class BaseNetDeployment:
    def __init__(self, models: list[BaseNetModel], preprocess=None, posprocess=None):
        self.preprocess = preprocess
        self.posprocess = posprocess
        self.models = {model.name: model for model in models}
        self.current_target = models[-1].name
        self.current_scope = self.update_scope()

    def update_scope(self, devices: (list, dict) = None):
        """
        This function updates the current computational scope.
        :param devices: The current devices to be used.
        :return: The current scope (tensorflow)
        """
        if devices is None:
            _devices = device_lib.list_local_devices()
        else:
            _devices = list(devices)
        strategy = tf.distribute.MirroredStrategy(devices=_devices)
        self.current_scope = strategy.scope()
        return self.current_scope

    def run(self, *args, **kwargs):
        with self.current_scope:
            preprocess_data = self.preprocess(*args, **kwargs['pre'])
            model_data = self.models[self.current_target].predict(preprocess_data, **kwargs['mod'])
            posprocess_data = self.posprocess(model_data, **kwargs['pos'])
        return posprocess_data

    def __repr__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
