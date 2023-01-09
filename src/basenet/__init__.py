# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
"""
.. include:: ../../README.md
"""
# import json
# import os
# from .__special__ import __config_path__
# with open(__config_path__, 'r', encoding='utf-8') as file:
#     cfg = json.load(file)
#     os.environ["CUDA_VISIBLE_DEVICES"] = cfg['gpu_devices']
from ._names import PREBUILT_LOSSES, KERAS_LIST_LAYERS, PREBUILT_LAYERS, KERAS_LOSSES, \
    KERAS_OPTIMIZERS, PREBUILT_OPTIMIZERS
from basenet.__special import window_diff
from .__special__ import __version__

from .deeplearning import BaseNetCompiler, BaseNetResults, BaseNetModel, BaseNetFeeder
from .metaheuristic import BaseNetHeuristic, BaseNetRandomSearch, BaseNetGenetic
from .classic import BaseNetLMSE
from .database import BaseNetDatabase
BaseNetCompiler.set_up_devices()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
