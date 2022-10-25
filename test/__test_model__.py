# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
from basenet import BaseNetCompiler, BaseNetDatabase
import pickle
# -----------------------------------------------------------


def test():
    with open('./db/testdb.ht', 'rb') as file:
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


def test2():
    basenet_model = BaseNetCompiler.build_from_yaml(verbose=True).compile()
    basenet_model.add_database(db_path='./db/mydb.db')
    results = basenet_model.fit(0, 3, tensorboard=False, avoid_lock=True)

    current_results = {}
    while results.is_training:
        current_results = results.get()
    basenet_model.recover()

    print(current_results)


if __name__ == '__main__':
    # test()
    test2()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
