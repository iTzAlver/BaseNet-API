# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class Generator:
    """Generator class creates a generator for a database."""
    def __init__(self, **kwargs):
        self.is_valid = False
        self.sym = True
        self.add(**kwargs)

    def __repr__(self):
        _text = f'Generator object with the arguments:\n{self.__dict__.__repr__()}\n'
        if self.is_valid:
            _text = f'{_text}The Generator is valid.'
        else:
            _text = f'{_text}The Generator is NOT valid.'
        return _text

    def __bool__(self):
        return self.is_valid

    def add(self, **kwargs):
        self.__dict__.update(kwargs)
        keys = kwargs.keys()
        if 'path' in keys and 'distribution' in keys and 'tput' in keys and 'awgn_m' in keys \
                and 'awgn_v' in keys and 'off_m' in keys and 'off_v' in keys and 'clust_m' in keys \
                and 'clust_v' in keys and 'number' in keys and 'name' in keys:
            self.is_valid = True

    def __getitem__(self, item):
        return self.__dict__[item]
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
