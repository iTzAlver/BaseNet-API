# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
from basenet import BaseNetDatabase


# -----------------------------------------------------------
def test_eq_add_split() -> None:
    original = BaseNetDatabase.load('./wikipedia_checkpoint.db')
    duplicated = original + original
    tuples_original = duplicated / 2
    assert tuples_original[0] == original
    assert tuples_original[1] == original
# -----------------------------------------------------------
# Main:


if __name__ == '__main__':
    test_eq_add_split()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
