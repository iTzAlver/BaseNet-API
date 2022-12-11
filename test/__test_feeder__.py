# -------------------------------------------------------------------#
#                                                                    #
#    Author:    Alberto Palomo Alonso.                               #
#                                                                    #
#    Git user:  https://github.com/iTzAlver                          #
#    Email:     ialver.p@gmail.com                                   #
#                                                                    #
# -------------------------------------------------------------------#
from src.basenet.feeder import BaseNetFeeder
import numpy as np


# -------------------------------------------------------------------#
#   x   x   x   x   x   x   x   x   x   x   x   x   x   x   x   x    #
# -------------------------------------------------------------------#
def feed(number_of_samples):
    x = list()
    y = list()
    for _ in range(number_of_samples):
        this_x = np.random.randint(0, 12, 10)
        n = np.random.randint(0, 10)
        x.append(np.append(this_x, n))
        y.append(np.concatenate([this_x[-n:], this_x[:-n]], 0))
    return np.array(x), np.array(y)


def main() -> None:
    my_feeder = BaseNetFeeder(feed, input_parameters=(10,))
    xtrain = my_feeder.xtrain
    xtrain2 = my_feeder.xtrain
    my_feeder.refresh()
    xtrain3 = my_feeder.xtrain
    print(xtrain == xtrain2)
    print(xtrain2 == xtrain3)


# -------------------------------------------------------------------#
#   x   x   x   x   x   x   x   x   x   x   x   x   x   x   x   x    #
# -------------------------------------------------------------------#
if __name__ == '__main__':
    main()
# -------------------------------------------------------------------#
#           E   N   D          O   F           F   I   L   E         #
# -------------------------------------------------------------------#
