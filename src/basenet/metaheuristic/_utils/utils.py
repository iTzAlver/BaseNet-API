# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
KEYWORD = 'must'


# -----------------------------------------------------------
def get_algorithm_parameters(**args):
    """
    This function returns the algorithm parameters.
    :param args: The arguments, including the kwargs of the constructor.
    :return: A tuple with the parameters.
    """
    kwargs = args.pop('kwargs')
    __return_values = list()
    for key, item in args.items():
        if key in kwargs:
            __return_values.append(kwargs.pop(key))
        else:
            if item == KEYWORD:
                raise ValueError(f'Error in getting the algorithm parameters: the value "{key}" must be included.')
            else:
                __return_values.append(item)
    __return_values.append(kwargs)
    return tuple(__return_values)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
