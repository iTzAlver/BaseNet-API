# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import random
import numpy as np
import pickle


# -----------------------------------------------------------
class BaseNetDatabase:
    def __init__(self, x, y, distribution: dict = None, name='unnamed_database', batch_size: int = None,
                 rescale: float = 1.0, dtype: str = 'float', bits: int = 32):
        """
        This class builds a BaseNetDatabase, compatible with the NetBase API.
        :param x: Inputs of the dataset.
        :param y: Solutions of the dataset.
        :param distribution: The distribution of the datasets, default: {'train': 70, 'val': 20, 'test': 10}
        :param name: The database name.
        :param batch_size: Custom batch size for training.
        :param rescale: Rescale factor, all the values in x are divided by this factor.
        :param dtype: Data type of the dataset.
        :param bits: Bits used for the data type.
        """
        if distribution is None:
            _distribution = (70, 20, 10)
        else:
            _distribution = (distribution['train'], distribution['val'], distribution['test'])

        self.name: str = name
        self.distribution: distribution

        (xtrain, ytrain), (xtest, ytest), (xval, yval) = self._splitdb((self._rescale(x, rescale), y),
                                                                       _distribution)

        self.dtype = f'{dtype}{bits}'
        self.xtrain = np.array(xtrain, dtype=self.dtype)
        self.ytrain = np.array(ytrain, dtype=self.dtype)
        self.xval = np.array(xval, dtype=self.dtype)
        self.yval = np.array(yval, dtype=self.dtype)
        self.xtest = np.array(xtest, dtype=self.dtype)
        self.ytest = np.array(ytest, dtype=self.dtype)

        if batch_size is None:
            self.batch_size = 2 ** round(np.log2(len(xtrain) / 256))
        else:
            self.batch_size = batch_size

    @staticmethod
    def load(path):
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
            print(f'BaseNetDatabase: Failed to load {path}: {ex}')
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
                print(f'BaseNetDatabase: Failed to save {path}: the path does not exist.')
                return False
        except Exception as ex:
            print(f'BaseNetDatabase: Failed to save {path}: {ex}')
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
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
