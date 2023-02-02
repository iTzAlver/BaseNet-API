# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
from ..database import BaseNetDatabase
from ..deeplearning import BaseNetFeeder, BaseNetModel
from PIL import Image
import numpy as np
# -----------------------------------------------------------


class BaseNetCVVisualizer:
    def __init__(self, database: (BaseNetDatabase, BaseNetFeeder), model: BaseNetModel = None):
        self.current_access = 'test'
        self.model = model
        self.feeder = None
        self.database = None
        self.preprocess = lambda x: x
        self.feeding_type = 'unknown'
        self.link_database(database)

    def link_database(self, database: (BaseNetDatabase, BaseNetFeeder)):
        if isinstance(database, BaseNetDatabase):
            self.database = database
            self.feeder = None
            self.feeding_type = 'from_db'
        elif isinstance(database, BaseNetFeeder):
            self.feeder = database
            self.database = None
            self.feeding_type = 'from_feeder'
        else:
            raise ValueError(f'BaseNetCVVisualizer: Cannot import type {type(database)} in a image visualizer.')
        return self

    def link_model(self, model: BaseNetModel):
        if isinstance(model, BaseNetModel):
            if not model.is_trained:
                logging.warning('BaseNEtCVVisualizer: The current model is not trained, you should have provided a'
                                'trained model.')
            self.model = model
        else:
            raise ValueError(f'BaseNetCVVisualizer: Cannot import the model because it is type {type(model)} and it'
                             f'is not compatible with the framework, please provide a BaseNetModel.')
        return self

    def check_compatibility(self) -> bool:
        if self.model is None:
            logging.warning('BaseNetCVVisualizer: cannot check compatibility without a linked model. '
                            'Please, link a model first.')
            return False
        if self.database is None and self.feeder is None:
            logging.warning('BaseNetCVVisualizer: cannot check compatibility without a linked database. '
                            'Please, link a database first.')
            return False
        io_shape = self.model.compiler.io_shape
        if self.database is not None:
            shape_x = self.preprocess(getattr(self.database, f'x{self.current_access}')[0]).shape
            shape_y = self.preprocess(getattr(self.database, f'y{self.current_access}')[0]).shape[0]
        else:
            shape_x = self.preprocess(getattr(self.feeder, f'x{self.current_access}')[0]).shape
            shape_y = self.preprocess(getattr(self.feeder, f'y{self.current_access}')[0]).shape[0]

        if not shape_x == io_shape[0]:
            logging.warning('BaseNetCVVisualizer: The input dimension of the model is not compatible with the input '
                            f'dimension of the dataset. Database has {shape_x} shape and the model has {io_shape[0]}.')
            return False
        if not shape_y == io_shape[1]:
            logging.warning('BaseNetCVVisualizer: The output dimension of the model is not compatible with the output '
                            f'dimension of the dataset. Database has {shape_y} shape and the model has {io_shape[1]}.')
            return False

        if not (2 <= len(shape_x) <= 3):
            logging.error(f'BaseNetCVVisualizer: The input database are not images or image-like data. Its shape is '
                          f'{shape_x}, it should have 2 (for black and white images) or 3 (for RGB-like images) fields, '
                          f'not {len(shape_x)}. How is that supposed to be an image? Re-format your database...')
            return False
        return True

    def set_access(self, access_type: str):
        if 'al' in access_type:
            self.current_access = 'val'
        elif 'est' in access_type:
            self.current_access = 'test'
        elif 'rain' in access_type:
            self.current_access = 'train'
        else:
            raise ValueError(f'BaseNetCVVisualizer: Cannot import {access_type} in the access type, '
                             f'use "train", "val" or "test".')
        return self

    def define_preprocess(self, function_like):
        self.preprocess = function_like
        return self

    def get(self, scale: float = 255) -> Image:
        if self.check_compatibility():
            arrays = getattr(self.database, f'x{self.current_access}')
            index = np.random.randint(arrays.shape[0])
            array = arrays[index]
            label = getattr(self.database, f'y{self.current_access}')[index]
            if len(array.shape) == 3:
                return Image.fromarray((array * scale).astype(np.uint8)).convert('RGB'), self.database.map([label]), array
            else:
                return Image.fromarray(array * scale), self.database.map([label]), array
        else:
            return None
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
