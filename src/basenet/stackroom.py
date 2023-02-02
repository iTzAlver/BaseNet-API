# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import os
import yaml
from .__special__ import __version__

INITIALIZE_CATEGORICAL = {'class_balance': list(),  'is_categorical': True, 'label_map': None}
INITIALIZE_NON_CATEGORICAL = {'statistical_balance': list(), 'is_categorical': False}


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        MAIN CLASS                         #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class BaseNetStackRoom:
    def __init__(self, room_path: str, batch_size: int = None, patch_size: int = None, name: str = 'default_bnsr_name',
                 categorical: bool = False):
        """
        """
        self.__version__ = __version__
        self.room_path = room_path
        self.__info_path = f'{room_path}/.info.yaml'

        if not os.path.exists(self.room_path):
            os.mkdir(self.room_path)
        if os.path.exists(self.__info_path):
            with open(self.__info_path, 'rb') as file:
                self.config = yaml.load(file, yaml.SafeLoader)
        else:
            if categorical:
                self.database_info = INITIALIZE_CATEGORICAL
            else:
                self.database_info = INITIALIZE_NON_CATEGORICAL
            self.config = {'batch_size': batch_size, 'patch_size': patch_size, 'name': name,
                           'database_data_type': tuple(), 'database_shape': tuple(), 'number_of_databases': 0,
                           'database_information': self.database_info}
            self.__create_stack_room()

    def __create_stack_room(self):
        with open(self.__info_path, 'wb') as file:
            yaml.dump(self.config, file, yaml.SafeDumper)

    def __repr__(self):
        return ""


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                           MAIN                            #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
if __name__ == '__main__':
    main()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
