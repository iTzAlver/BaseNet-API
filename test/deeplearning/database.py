# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
from messages import do_assert, LowLevelError
from basenet import BaseNetDatabase
import random
import os

IO_PATH = './my_database.db'


# -----------------------------------------------------------
def basenet_database_test(logger, preamble):
    """
    TEST:
    ----------
    BaseNetDatabase
    ----------

    :param logger: The top level logger.
    :param preamble: The top level message.
    :return: Nothing.
    """
    # Test 0:
    def create_random_dataset(x_dim, y_dim):
        return [[random.random() for _ in range(x_dim)] for _ in range(y_dim)], [random.randint(0, 10) for _ in
                                                                                 range(y_dim)]

    x, y = create_random_dataset(3, 10)
    _y_db = BaseNetDatabase(x, y,
                            distribution={'train': 60, 'val': 20, 'test': 20},
                            name='testings',
                            batch_size=2,
                            rescale=2,
                            dtype=('float', 'int'),
                            bits=(32, 8))

    do_assert(_y_db.size, (6, 2, 2), LowLevelError.construction_raw)
    do_assert(_y_db.name, 'testings', LowLevelError.construction)
    do_assert(_y_db.batch_size, 2, LowLevelError.construction)
    do_assert(_y_db.dtype, ('float32', 'int8'), LowLevelError.construction)
    do_assert(_y_db.distribution, (60, 20, 20), LowLevelError.construction)
    do_assert(_y_db.is_valid, True, LowLevelError.construction_raw)

    x, y = create_random_dataset(3, 3000)
    my_db = BaseNetDatabase(x, y,
                            distribution={'train': 60, 'val': 20, 'test': 20},
                            name='testings',
                            dtype=('float', 'int'),
                            bits=(32, 8))

    do_assert(my_db.batch_size, 8, LowLevelError.auto_batch)

    my_db.save(IO_PATH)
    if os.path.exists(IO_PATH):
        my_db_load = BaseNetDatabase.load(IO_PATH)
        do_assert(my_db, my_db_load, LowLevelError.importing)
        os.remove(IO_PATH)
    else:
        do_assert(os.path.exists(IO_PATH), True, LowLevelError.exporting)

    _split = _y_db / 2
    do_assert(_split, _y_db.split(2), LowLevelError.split)
    do_assert(len(_split), 2, LowLevelError.split)
    do_assert(_split[0].size, (3, 1, 1), LowLevelError.split)

    _merge = _split[0] + _split[1]
    do_assert(_merge, _y_db, LowLevelError.merge)
    do_assert(_merge.size, (6, 2, 2), LowLevelError.merge)

    train_x = my_db.xtrain
    train_y = my_db.ytrain
    val_x = my_db.xval
    val_y = my_db.yval
    test_x = my_db.xtest
    test_y = my_db.ytest
    my_manual_db = BaseNetDatabase.from_datasets(train=(train_x, train_y), val=(val_x, val_y), test=(test_x, test_y))
    do_assert(my_manual_db.size, my_db.size, LowLevelError.explicit)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
