# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
from basenet import BaseNetLMSE, BaseNetDatabase, window_diff

WORKBENCH = 64
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
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
