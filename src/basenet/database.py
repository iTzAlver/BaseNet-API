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
import tensorflow as tf
import logging
import random
import numpy as np
import pickle
import copy
import pandas as pd

from .__special__ import __version__


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
    def __init__(self, x, y=None, distribution: dict = None, name='unnamed_database', batch_size: int = None,
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
        self.__version__ = __version__
        try:
            if y is None or isinstance(y, str):
                x_, y_ = self._framework_convertion(x, y)
            else:
                x_, y_ = x, y

            if distribution is None:
                _distribution = (70, 20, 10)
            else:
                _distribution = (distribution['train'], distribution['val'], distribution['test'])

            self.name: str = name
            self.distribution = _distribution

            if isinstance(y_, np.ndarray):
                _y = copy.copy(y_).tolist()
            else:
                _y = copy.copy(y_)

            _y = self._to_binary(_y)
            _x = self._rescale(copy.copy(x_), rescale)

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
                self.batch_size = 1
                if len(xtrain) > 0:
                    self.batch_size = 2 ** round(np.log2(len(xtrain) / 256))
                if self.batch_size < 1:
                    self.batch_size = 1
            else:
                self.batch_size = batch_size

            self.is_valid = False
            if sum(_distribution) == 100:
                self._check_validation()
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
                if hasattr(self, '__version__'):
                    if __version__ == self.__version__:
                        return self
                return BaseNetDatabase._reversion(self)
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

    def split(self, other):
        """
        The split function of the BaseNetDatabase divides a BaseNetDatabase in n parts.
        :param other: The number of parts to divide the database.
        :return: A tulpe of splitted BaseNetDatabases.
        """
        if isinstance(other, int):
            self._check_validation()
            if other > 0:
                if self.is_valid:
                    try:
                        return self._split(other)
                    except Exception as ex:
                        logging.error(f'BaseNetDatabase: Exception raised while splitting two BaseNetDatabase: {ex}')
                        return self
                else:
                    logging.error(f'BaseNetDatabase: The current BaseNetDatabase is not a valid database. '
                                  f'The module is returning a non splitted BaseNetDatabase. ')
                    return self
            else:
                logging.error(f'BaseNetDatabase: Trying to divide the database in a number lower than 0. '
                              f'The module is returning a non splitted BaseNetDatabase.')
                return self
        else:
            logging.error(f'BaseNetDatabase: Trying to divide the database in something different to an integer. '
                          f'The module is returning a non splitted BaseNetDatabase. '
                          f'Expecting type "int", given {type(other)}.')
            return self

    def merge(self, other):
        """
        The merge function of the BaseNetDatabase Class merges two BaseNetDatabases.
        :param other: A BaseNetDatabase object to merge.
        :return: A merged BaseNetDatabase.
        """
        if isinstance(other, BaseNetDatabase):
            self._check_validation()
            other._check_validation()
            if other:
                if self.is_valid:
                    try:
                        return self._merge(other)
                    except Exception as ex:
                        logging.error(f'BaseNetDatabase: Exception raised while merging two BaseNetDatabase: {ex}')
                        return self
                else:
                    logging.error(f'BaseNetDatabase: The current BaseNetDatabase is not a valid database. '
                                  f'The module is returning a non merged BaseNetDatabase. ')
                    return self
            else:
                logging.error(f'BaseNetDatabase: The incoming BaseNetDatabase is not a valid database. '
                              f'The module is returning a non merged BaseNetDatabase. ')
                return self
        else:
            logging.error(f'BaseNetDatabase: Trying to merge the database in something different to a BaseNetDatabase. '
                          f'The module is returning a non merged BaseNetDatabase. '
                          f'Expecting type "BaseNetDatabase", given {type(other)}.')
            return self

    # Private methods:
    @staticmethod
    def _reversion(input_db):
        reversioned = BaseNetDatabase([], [])
        reversioned.xtrain = input_db.xtrain
        reversioned.xval = input_db.xval
        reversioned.xtest = input_db.xtest
        reversioned.ytrain = input_db.ytrain
        reversioned.yval = input_db.yval
        reversioned.ytest = input_db.ytest
        reversioned.size = input_db.size
        reversioned.name = input_db.name
        reversioned.dtype = input_db.dtype
        reversioned.distribution = input_db.distribution
        reversioned._check_validation()
        return reversioned

    def _check_validation(self):
        if sum(self.size) > 0:
            c_train = len(self.xtrain) == len(self.ytrain)
            c_test = len(self.xtest) == len(self.ytest)
            c_val = len(self.xval) == len(self.yval)
            if c_test and c_train and c_val:
                self.is_valid = True
            else:
                self.is_valid = False
        else:
            self.is_valid = False

    @staticmethod
    def _splitdb(setz: tuple, split: tuple) -> tuple:
        # This function splits the database into test, train and validation from a single distribution.
        total = len(setz[0])
        xtrain = list()
        ytrain = list()
        xval = list()
        yval = list()
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

    def _split(self, other):
        splits_train = np.linspace(0, self.size[0], other + 1)
        spltis_val = np.linspace(0, self.size[1], other + 1)
        splits_test = np.linspace(0, self.size[2], other + 1)
        last_index_train = 0
        last_index_val = 0
        last_index_test = 0
        list_of_dbs = list()
        for _split_train, _split_val, _split_test in zip(splits_train[1:], splits_test[1:], spltis_val[1:]):
            split_train = int(np.ceil(_split_train))
            split_val = int(np.ceil(_split_val))
            split_test = int(np.ceil(_split_test))
            this_db = copy.copy(self)
            this_db.xtrain = self.xtrain[last_index_train:split_train]
            this_db.xval = self.xval[last_index_val:split_val]
            this_db.xtest = self.xtest[last_index_test:split_test]
            this_db.ytrain = self.ytrain[last_index_train:split_train]
            this_db.yval = self.yval[last_index_val:split_val]
            this_db.ytest = self.ytest[last_index_test:split_test]
            this_db.size = (len(this_db.xtrain), len(this_db.xval), len(this_db.xtest))
            this_db.distribution = (len(this_db.xtrain) / sum(this_db.size), len(this_db.xval) / sum(this_db.size),
                                    len(this_db.xtest) / sum(this_db.size))
            if this_db.size[0] < this_db.batch_size:
                this_db.batch_size = this_db.size[0]
                logging.warning(f'BaseNetDatabase: The splitted database size is lower than the initial batch size. '
                                f'Consider reasigning the batch_size attribute properly; now it is resized to the '
                                f'length of xtrain.')
            this_db._check_validation()
            list_of_dbs.append(this_db)
            last_index_train = split_train
            last_index_val = split_val
            last_index_test = split_test
        return tuple(list_of_dbs)

    def _merge(self, other):
        self.xtrain = np.append(self.xtrain, other.xtrain, axis=0)
        self.ytrain = np.append(self.ytrain, other.ytrain, axis=0)
        self.xval = np.append(self.xval, other.xval, axis=0)
        self.yval = np.append(self.yval, other.yval, axis=0)
        self.xtest = np.append(self.xtest, other.xtest, axis=0)
        self.ytest = np.append(self.ytest, other.ytest, axis=0)
        self.size = (len(self.xtrain), len(self.xval), len(self.xtest))
        self.distribution = (len(self.xtrain) / sum(self.size), len(self.xval) / sum(self.size),
                             len(self.xtest) / sum(self.size))
        self._check_validation()
        return self

    @staticmethod
    def _to_binary(_y):
        """
        This function can convert:
        [0, 3, 5, 6, 9, 4, 9] -> [[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0]] (binarized)
        [5.67, 0.92, 0.12, 6.32] -> [0.8, 0.1, 0.12, 1]                                     (normalized)
            Untouched:
        [0.8, 0.1, 0.3, 0.1, 0.8] -> [0.8, 0.1, 0.3, 0.1, 0.8]
            Error:
        []              -> ValueError
        Object (false): -> ValueError
        """
        try:
            __y = np.array(_y)
            if len(__y.shape) == 2:
                return _y
            if not _y:
                raise ValueError(f'BaseNetDatabaseError: The value of y is not provided.')
            else:

                my = max(__y)
                if __y.dtype != int:
                    if my > 1:
                        return (__y / my).tolist()
                    else:
                        return _y
                else:
                    y = list()
                    for element in _y:
                        element_y = np.zeros(my + 1)
                        element_y[element] = 1
                        y.append(element_y)
            return y
        except Exception as ex:
            raise RuntimeError('BaseNetDatabase:_to_binary: An error ocurred while converting the non-binzarized and'
                               f'non-normalized labels into the API database:\n{ex}')

    @staticmethod
    def _framework_convertion(x, y_hint=None):
        if isinstance(x, pd.DataFrame):
            if y_hint is None:
                _x = np.array(x)
                return _x[:, :-1], _x[:, -1]
            else:
                try:
                    _y = np.array(x[y_hint])
                    _x = np.array(x.loc[:, x.columns != y_hint])
                    return _x, _y
                except Exception as ex:
                    raise ValueError('BaseNetDatabase Error: The given value of y is not a valid column for the given'
                                     f'DataFrame.\n{ex}')
        elif isinstance(x, tf.data.Dataset):
            _x = list()
            _y = list()
            for batched in x:
                _x.append(batched['image'])
                _y.append(batched['label'])
            return _x, _y
        else:
            raise ValueError('BaseNetDatabase Error: The values of y are not provided and the input x is not '
                             'recognized as a compatible framework.')

    def __bool__(self):
        return self.is_valid

    def __repr__(self):
        return f'BaseNetDatabase with {sum(self.size)} instances.'

    def __call__(self, *args, **kwargs):
        """
        The call function of the BaseNetDatabase Class merges two BaseNetDatabases.
        :param args: A BaseNetDatabase object to merge.
        :param kwargs: Ignored.
        :return: A merged BaseNetDatabase
        """
        return self.merge(args[0])

    def __add__(self, other):
        """
        The add function of the BaseNetDatabase Class merges two BaseNetDatabases.
        :param other: A BaseNetDatabase object to merge.
        :return: A merged BaseNetDatabase.
        """
        copy_object = copy.copy(self)
        return copy_object.merge(other)

    def __truediv__(self, other):
        """
        The add function of the BaseNetDatabase Class divides a BaseNetDatabase in n parts.
        :param other: The number of parts to divide the database.
        :return: A tulpe of splitted BaseNetDatabases.
        """
        copy_object = copy.copy(self)
        return copy_object.split(other)

    def __eq__(self, other):
        if isinstance(other, BaseNetDatabase):
            if self and other:
                if self.size == other.size:
                    cs_train = (self.xtrain.size == other.xtrain.size) and (self.ytrain.size == other.ytrain.size)
                    cs_val = (self.xval.size == other.xval.size) and (self.yval.size == other.yval.size)
                    cs_test = (self.xtest.size == other.xtest.size) and (self.ytest.size == other.ytest.size)
                    if cs_test and cs_val and cs_train:
                        c_train = (self.xtrain == other.xtrain).all() and (self.ytrain == other.ytrain).all()
                        c_val = (self.xval == other.xval).all() and (self.yval == other.yval).all()
                        c_test = (self.xtest == other.xtest).all() and (self.ytest == other.ytest).all()
                        if c_test and c_val and c_train:
                            return True
        return False
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
