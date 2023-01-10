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
NOISE = 0

NOISES = ['0', '01', '02', '03', '04', '05', '06']
DATABASE_PATH = f'C:/Users/ialve/CorNet/db/ht/sym/32k_{WORKBENCH}t_{NOISES[NOISE]}w.db'


# -----------------------------------------------------------
if __name__ == '__main__':
    database = BaseNetDatabase.load(DATABASE_PATH)
    lmse = BaseNetLMSE(database, name='test_model', th=0.5)
    results = lmse.evaluate(window_diff)
    _test_ = f'Results:\n\tMSE:\t{lmse.results["mse"]}\n\tMAE:\t{lmse.results["mae"]}\n\tError:\t' \
             f'{lmse.results["error"]}\n\tWD:\t\t{results}'
    print(_test_)
    trans = lmse.transformation()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = np.meshgrid(range(trans.shape[0]), range(trans.shape[1]), range(trans.shape[2]))
    sc = ax.scatter(x, y, z, c=trans.flat, cmap='inferno')
    plt.colorbar(sc)
    plt.show()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
