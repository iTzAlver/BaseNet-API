# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
import os.path
from basenet import BaseNetCompiler, BaseNetDatabase
import pickle
# -----------------------------------------------------------


def test():
    with open('db/testdb.ht', 'rb') as file:
        other_db = pickle.load(file)
    x = []
    x.extend(other_db.dataset.xtrain)
    x.extend(other_db.dataset.xval)
    x.extend(other_db.dataset.xtest)
    y = []
    y.extend(other_db.dataset.ytrain)
    y.extend(other_db.dataset.yval)
    y.extend(other_db.dataset.ytest)
    my_db = BaseNetDatabase(x, y, rescale=255)
    my_db.save('./db/mydb.db')
    assert os.path.exists('db/mydb.db')
    logging.info('Test 0 completed: Rework a database.')


def test1():
    basenet_model = BaseNetCompiler.build_from_yaml(verbose=True).compile()
    basenet_model.add_database(db_path='db/mydb.db')
    results = basenet_model.fit(0, 3, tensorboard=False, avoid_lock=False)
    current_results = results.get()
    print(current_results)
    logging.info('Test 1 completed: Single process fitting.')


def test2():
    basenet_model = BaseNetCompiler.build_from_yaml(verbose=True).compile()
    basenet_model.add_database(db_path='db/mydb.db')
    results = basenet_model.fit(0, 3, tensorboard=False, avoid_lock=True)

    current_results = {}
    while results.is_training:
        current_results = results.get()
    basenet_model.recover()

    print(current_results)
    logging.info('Test 2 completed: Multiprocessing fitting.')


def test3():
    basenet_model = BaseNetCompiler.build_from_yaml(verbose=True).compile()
    basenet_model.add_database(db_path='db/mydb.db')
    results = basenet_model.fit(0, 100, tensorboard=True, avoid_lock=False)
    current_results = results.get()
    print(current_results)
    logging.info('Test 3 completed: Tensorboard tested.')


if __name__ == '__main__':
    test()
    test1()
    test2()
    test3()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
