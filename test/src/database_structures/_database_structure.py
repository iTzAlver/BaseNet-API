# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
from ._dataset import Dataset
from ._generator import Generator
import random
import pickle


# -----------------------------------------------------------
class Database:
    def __init__(self, setz: tuple[list, list], generator: Generator, dbtype='hypertrain'):
        self.name: str = generator['name']
        self.distribution: dict = {'train': generator['distribution'][0],
                                   'validation': generator['distribution'][1],
                                   'test': generator['distribution'][2]}

        (xtrain, ytrain), (xtest, ytest), (xval, yval) = self._splitdb(setz, generator['distribution'])
        self.dataset: Dataset = Dataset(xtrain, ytrain, xtest, ytest, xval, yval)

        self.size: tuple[int, int, int] = (len(self.dataset.xtrain),
                                           len(self.dataset.xval),
                                           len(self.dataset.xtest))
        self.path: str = generator['path']
        self.type: str = dbtype
        self.batch_size = 64

    @staticmethod
    def _splitdb(setz: tuple[list, list], split: tuple):
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
            raise ValueError('Test size in Database class is too small.')
        return (xtrain, ytrain), (xtest, ytest), (xval, yval)

    def save(self):
        # Saves the model from the given path in the Generator.
        with open(self.path, 'wb') as file:
            pickle.dump(self, file)

    def __repr__(self):
        return f'Database structure with:\n' \
               f'\tTrain: {len(self.dataset.xtrain)}\n' \
               f'\tValidation: {len(self.dataset.xval)}\n'\
               f'\tTest: {len(self.dataset.xtest)}\n' \
               f'{self.distribution}.'

    @staticmethod
    def load(path):
        with open(path, 'rb') as file:
            self = pickle.load(file)
        return self

    def randget(self):
        types = random.randint(0, 2)
        if types == 0:
            tlen = len(self.dataset.xtrain)
            _type = 'Train'
            no = random.randint(0, tlen)
            mat = self.dataset.xtrain[no]
            ref = self.dataset.ytrain[no]
        elif types == 1:
            tlen = len(self.dataset.xval)
            _type = 'Validation'
            no = random.randint(0, tlen)
            mat = self.dataset.xval[no]
            ref = self.dataset.yval[no]
        else:
            tlen = len(self.dataset.xtest)
            _type = 'Test'
            no = random.randint(0, tlen)
            mat = self.dataset.xtest[no]
            ref = self.dataset.ytest[no]
        return _type, no, mat, ref
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
