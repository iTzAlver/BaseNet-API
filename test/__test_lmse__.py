# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import matplotlib.pyplot as plt
import numpy as np
from basenet import BaseNetLMSE, BaseNetDatabase, window_diff

WORKBENCH = 16
TRAIN_NOISE = 4
NOISE = 0

NOISES = ['0', '01', '02', '03', '04', '05', '06']
DATABASE_PATH = f'C:/Users/ialve/CorNet/db/ht/sym/32k_{WORKBENCH}t_{NOISES[NOISE]}w.db'
NOISELESS_PATH = f'C:/Users/ialve/CorNet/db/ht/sym/32k_{WORKBENCH}t_{NOISES[TRAIN_NOISE]}w.db'

# -----------------------------------------------------------
if __name__ == '__main__':
    database = BaseNetDatabase.load(NOISELESS_PATH)
    noisy_database = BaseNetDatabase.load(DATABASE_PATH)
    lmse = BaseNetLMSE(database, name='test_model', th=0.5)
    results = lmse.evaluate(window_diff)

    trans, bias = lmse.transformation(bias=True)
    true_trans, true_bias = lmse.transformation(original=False, bias=True)
    x, y = database.xtest[200], database.ytest[200]
    out = lmse.predict([x])

    noisy_lmse = BaseNetLMSE.load((true_trans, true_bias), name='noisy_model', th=0.5).link_database(noisy_database)
    noisy_lmse.validate()
    noisy_results = noisy_lmse.evaluate(window_diff)

    _test_ = f'Results:\n\tMSE:\t{lmse.results["mse"]}\n\tMAE:\t{lmse.results["mae"]}\n\tError:\t' \
             f'{lmse.results["error"]}\n\tWD:\t\t{results}'
    print(_test_)

    _test_ = f'\nResults (noisy):\n\tMSE:\t{noisy_lmse.results["mse"]}\n\tMAE:\t{noisy_lmse.results["mae"]}' \
             f'\n\tError:\t{noisy_lmse.results["error"]}\n\tWD:\t\t{noisy_results}'
    print(_test_)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    _x, _y, _z = np.meshgrid(range(trans.shape[0]), range(trans.shape[1]), range(trans.shape[2]))
    sc = ax.scatter(_x, _y, _z, c=trans.flat, cmap='inferno')
    plt.colorbar(sc)
    plt.show()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
