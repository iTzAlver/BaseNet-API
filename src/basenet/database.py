# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
"""
The database.py file contains the BaseNetDatabase class.
"""
# Import statements:
import logging
import random
import numpy as np
import pickle


# -----------------------------------------------------------
class BaseNetDatabase:
    """
    The BaseNetDatabase class converts a set of inputs (x) and solutions (y) into the API wrapper database.
    The BaseNetDatabase will create a new set of attributes from the database randomly:

        *   xtrain: A subset of (x) with the train inputs of the network.
        *   ytrain: A subset of (y) with the train solutions of the network.
        *   xval: A subset of (x) with the training validation inputs of the network.
        *   yval: A subset of (y) with the training validation solutions of the network.
        *   xtest: A subset of (x) with excluded inputs of the network; for future testing.
        *   ytest: A subset of (y) with excluded solutions of the network; for future testing.

        *   dtype: Data type of the input data (x) and output data (y) in a tuple of strings (x_dtype, y_dtype).
        *   name: Name of the database.
        *   distribution: Train, validation and test distribution of the input database.
        *   batch_size: Current batch size of the database.
        *   size: The size of the database (train, validation , test).

    The BaseNetDatabase can be loaded and saved with its own methods.
    """
    def __init__(self, x, y, distribution: dict = None, name='unnamed_database', batch_size: int = None,
                 rescale: float = 1.0, dtype: tuple[str, str] = ('float', 'float'), bits: tuple[int, int] = (32, 32)):
        """
        This class builds a BaseNetDatabase, compatible with the NetBase API.
        :param x: Inputs of the dataset.
        :param y: Solutions of the dataset.
        :param distribution: The distribution of the datasets, default: {'train': 70, 'val': 20, 'test': 10}
        :param name: The database name.
        :param batch_size: Custom batch size for training.
        :param rescale: Rescale factor, all the values in x are divided by this factor, in case rescale is needed.
        :param dtype: Data type of the dataset. ('input', 'output') (x, y)
        :param bits: Bits used for the data type. ('input', 'output') (x, y)
        """
        try:
            if distribution is None:
                _distribution = (70, 20, 10)
            else:
                _distribution = (distribution['train'], distribution['val'], distribution['test'])

            self.name: str = name
            self.distribution = _distribution

            if isinstance(y, np.ndarray):
                _y = y.tolist()
            else:
                _y = y

            _x = self._rescale(x, rescale)

            if len(_x) != len(_y):
                logging.error('BaseNetDatabase: Error while building the database, the number of instances of '
                              f'x and y must be the same. Found x: {len(_x)} != y: {len(y)}.')
                self.is_valid = False
                return

            (xtrain, ytrain), (xtest, ytest), (xval, yval) = self._splitdb((_x, _y), _distribution)

            self.dtype = (f'{dtype[0]}{bits[0]}', f'{dtype[1]}{bits[1]}')
            self.xtrain = np.array(xtrain, dtype=self.dtype[0])
            self.ytrain = np.array(ytrain, dtype=self.dtype[1])
            self.xval = np.array(xval, dtype=self.dtype[0])
            self.yval = np.array(yval, dtype=self.dtype[1])
            self.xtest = np.array(xtest, dtype=self.dtype[0])
            self.ytest = np.array(ytest, dtype=self.dtype[1])

            self.size = (len(self.xtrain), len(self.xval), len(self.xtest))

            if batch_size is None:
                self.batch_size = 2 ** round(np.log2(len(xtrain) / 256))
                if self.batch_size < 1:
                    self.batch_size = 1
            else:
                self.batch_size = batch_size

            if sum(_distribution) == 100:
                self.is_valid = True
            else:
                logging.warning('BaseNetDatabase: The sum of the distributions for train, validation and test does not '
                                'add up to 100%')
                self.is_valid = False
        except Exception as ex:
            self.is_valid = False
            logging.error(f'BaseNetDatabase: Error while building the database, raised the following exception: {ex}')

    @staticmethod
    def load(path: str):
        """
        This function loads the BaseNetDatabase from any path.
        :param path: Path where the BaseNetDatabase is being saved in the file system.
        :return: The loaded database if successful. 'None' if not.
        """
        try:
            if path:
                with open(path, 'rb') as file:
                    self = pickle.load(file)
                return self
            else:
                return None
        except Exception as ex:
            logging.error(f'BaseNetDatabase: Failed to load {path}: {ex}')
            return None

    def save(self, path: str):
        """
        This function saves the BaseNetDatabase in any format.
        :param path: Path where the BaseNetDatabase is being saved in the file system.
        :return: True if the saving was successful. False if not.
        """
        try:
            if path:
                with open(path, 'wb') as file:
                    pickle.dump(self, file)
                return True
            else:
                logging.warning(f'BaseNetDatabase: Failed to save {path}: the path does not exist.')
                return False
        except Exception as ex:
            logging.error(f'BaseNetDatabase: Failed to save {path}: {ex}')
            return False

    # Private methods:
    @staticmethod
    def _splitdb(setz: tuple, split: tuple) -> tuple:
        # This function splits the database into test, train and validation from a single distribution.
        total = len(setz[0])
        xtrain = []
        ytrain = []
        xval = []
        yval = []
        ntrain = round(total * split[0] / 100)
        nval = round(total * split[1] / 100)
        ntest = total - ntrain - nval
        if ntest >= 0:
            for _ in range(ntrain):
                topop = random.randint(0, len(setz[0]) - 1)
                xtrain.append(setz[0].pop(topop))
                ytrain.append(setz[1].pop(topop))
            for _ in range(nval):
                topop = random.randint(0, len(setz[0]) - 1)
                xval.append(setz[0].pop(topop))
                yval.append(setz[1].pop(topop))
            xtest = setz[0]
            ytest = setz[1]
        else:
            raise ValueError('BaseNetDatabase: Test size in BaseNetDatabase class is too small.')
        return (xtrain, ytrain), (xtest, ytest), (xval, yval)

    @staticmethod
    def _rescale(x, scale):
        return list(np.array(x) / scale)

    def __bool__(self):
        return self.is_valid

    def __repr__(self):
        return f'BaseNetDatabase with {sum(self.size)} instances.'
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
