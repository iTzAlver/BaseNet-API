# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import os.path
from basenet import BaseNetDatabase
import numpy as np
import logging


# -----------------------------------------------------------
def test_beech():
    # Test building 0.
    test0()
    # Test building 1.
    test1()
    # Test saving.
    test2()
    # Test loading.
    test3()


def test0():
    x = np.ones((10, 10), dtype=np.int32) * 255
    y = np.ones((10, 1), dtype=np.float32) * 0.5
    db0 = BaseNetDatabase(x, y)
    db1 = BaseNetDatabase(x, y, distribution={'train': 30, 'debug': 30, 'val': 40}, batch_size=64)
    db2 = BaseNetDatabase(x, y, distribution={'train': 30, 'debug': 30, 'val': 40}, batch_size=64, rescale=255,
                          dtype=('int', 'float'))
    assert db0
    assert db1
    assert db2
    logging.info('Test 0 completed: valid building.')


def test1():
    x = np.random.random((100, 100))
    y = np.random.random((1, 100))
    db0 = BaseNetDatabase(x, y)
    y = np.random.random((100, 1))
    db1 = BaseNetDatabase(y, x)
    assert not db0
    assert db1
    logging.info('Test 1 completed: invalid building.')


def test2():
    x = np.ones((10, 10), dtype=np.int32) * 255
    y = np.ones((10, 1), dtype=np.float32) * 0.5
    db = BaseNetDatabase(x, y, distribution={'train': 30, 'debug': 30, 'val': 40}, batch_size=64, rescale=255,
                         dtype=('int', 'float'))
    db.save('./testdir/testmodel_0.db')
    assert os.path.exists('testdir/testmodel_0.db')
    logging.info('Test 2 completed: saving.')


def test3():
    x = np.ones((10, 10), dtype=np.int32) * 255
    y = np.ones((10, 1), dtype=np.float32) * 0.5
    db0 = BaseNetDatabase(x, y, distribution={'train': 30, 'debug': 30, 'val': 40}, batch_size=64, rescale=255,
                          dtype=('int', 'float'))
    db1 = BaseNetDatabase.load('./testdir/testmodel_0.db')
    for attribute in ['xtrain', 'ytrain', 'xval', 'yval', 'xtest', 'ytest']:
        for nr, row in enumerate(getattr(db0, attribute)):
            for nc, element in enumerate(row):
                assert getattr(db1, attribute)[nr, nc] == element
    logging.info('Test 3 completed: loading.')


# -----------------------------------------------------------
if __name__ == '__main__':
    test_beech()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
