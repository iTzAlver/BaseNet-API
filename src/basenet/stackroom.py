# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
import os
import yaml
import shutil
import numpy as np
from .database import BaseNetDatabase
from .__special__ import __version__

INITIALIZE_CATEGORICAL = {'class_count': list(), 'class_balance': dict(), 'is_categorical': True,
                          'label_map': None}
INITIALIZE_NON_CATEGORICAL = {'statistical_count': list(), 'statistical_balance': dict(),
                              'is_categorical': False, 'fuzzy_map': None}


__config_file__ = 'info.yaml'
__storage_dir__ = 'stackroom'
__test_dir__ = 'tests'
__model_dir__ = 'models'


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        MAIN CLASS                         #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class BaseNetStackRoom:
    def __init__(self, room_path: str, batch_size: int = None, patch_size: float = None,
                 name: str = 'default_bnsr_name', categorical: bool = False):
        """
        """
        self.__version__ = __version__
        self.room_path = room_path
        self.__info_path = f'{room_path}/{__config_file__}'
        self.current_index_train = 0
        self.current_index_test = 0
        self.__is_built = False

        if not os.path.exists(self.room_path):
            os.mkdir(self.room_path)
        if os.path.exists(self.__info_path):
            with open(self.__info_path, 'r', encoding='utf-8') as file:
                self.config = yaml.load(file, yaml.Loader)
        else:
            if categorical:
                self.database_info = INITIALIZE_CATEGORICAL
            else:
                self.database_info = INITIALIZE_NON_CATEGORICAL
            self.config = {'batch_size': batch_size, 'patch_size': patch_size, 'name': name,
                           'size': {'train': (0, 0, 0), 'test': (0, 0, 0)},
                           'database_data_type': tuple(), 'database_shape': tuple(), 'number_of_databases': 0,
                           'database_information': self.database_info, 'current_sizes_mb': dict(), 'current_size_mb': 0}
            self.update_info()

    def get(self, train: bool = True):
        if not self.__is_built:
            logging.error('BaseNetStackRoom: Unable to access data because the current StackRoom is not created yet.')
            return None
        if train:
            all_ones = os.listdir(f'{self.room_path}/{__storage_dir__}')
            dbsel = None
            for _ in all_ones:
                if int(_.split('_')[0]) == self.current_index_train:
                    dbsel = _
                    break
            db = BaseNetDatabase.load(f'{self.room_path}/{__storage_dir__}/{dbsel}')
            if self.current_index_train >= len(all_ones) - 1:
                self.current_index_train = 0
            else:
                self.current_index_train += 1
        else:
            all_ones = os.listdir(f'{self.room_path}/{__test_dir__}')
            dbsel = None
            for _ in all_ones:
                if int(_.split('_')[0]) == self.current_index_test:
                    dbsel = _
                    break
            db = BaseNetDatabase.load(f'{self.room_path}/{__storage_dir__}/{dbsel}')
            if self.current_index_test >= len(all_ones) - 1:
                self.current_index_test = 0
            else:
                self.current_index_test += 1
        return db

    def update_info(self):
        with open(self.__info_path, 'w', encoding='utf-8') as file:
            yaml.dump(self.config, file, default_flow_style=False, encoding='utf-8')

    def add_database(self, database: BaseNetDatabase, train: bool = True):
        if train:
            if database.xtest:
                database.xtrain = np.concatenate([database.xtrain, database.xtest])
                database.ytrain = np.concatenate([database.ytrain, database.ytest])
            database.xtest = np.array([database.xtrain[0]])
            database.ytest = np.array([database.ytrain[0]])
            database.size = (len(database.xtrain), len(database.xval), len(database.xtest))
            ret_value = self.__add_database_train(database)
            self.config['size']['train'] = (database.size[0] + self.config['size']['train'][0],
                                            database.size[1] + self.config['size']['train'][1],
                                            database.size[2] + self.config['size']['train'][2])
            if not ret_value and self.config['number_of_databases'] == 0:
                shutil.rmtree(f'{self.room_path}')
        else:
            if database.xtrain:
                database.xtest = np.concatenate([database.xtest, database.xtrain])
                database.ytest = np.concatenate([database.ytest, database.ytrain])
            database.ytrain = np.array([database.ytest[0]])
            database.xtrain = np.array([database.xtest[0]])
            if database.xval:
                database.xtest = np.concatenate([database.xtest, database.xval])
                database.ytest = np.concatenate([database.ytest, database.yval])
            database.xval = np.array([database.xtest[0]])
            database.yval = np.array([database.ytest[0]])
            database.size = (len(database.xtrain), len(database.xval), len(database.xtest))
            self.config['size']['test'] = (database.size[0] + self.config['size']['test'][0],
                                           database.size[1] + self.config['size']['test'][1],
                                           database.size[2] + self.config['size']['test'][2])
            ret_value = self.__add_database_test(database)
        self.update_info()
        return ret_value

    def __add_database_test(self, database: BaseNetDatabase):
        if self.__compute_patch_size(database) > self.config['patch_size']:
            logging.error('BaseNetStackRoom: Cannot import the database because it exceeds the patch size.')
            return False
        else:
            nod = len(os.listdir(f'{self.room_path}/{__test_dir__}'))
            database.name = f'{nod}_{database.name}'
            database.save(f'{self.room_path}/{__test_dir__}/{database.name}.db')
            return True

    def __add_database_train(self, database: BaseNetDatabase):
        if self.config['number_of_databases'] == 0 or not os.path.exists(f'{self.room_path}/{__storage_dir__}'):
            os.mkdir(f'{self.room_path}/{__storage_dir__}')
            os.mkdir(f'{self.room_path}/{__test_dir__}')
            os.mkdir(f'{self.room_path}/{__model_dir__}')
            self.config['number_of_databases'] = 0
            self.config['database_data_type'] = database.dtype
            self.config['database_shape'] = database.shape
            if self.config['batch_size'] is None:
                self.config['batch_size'] = database.batch_size
            else:
                database.batch_size = self.config['batch_size']
            if self.config['name'] is None:
                self.config['name'] = database.name
            initial_current_size = self.__compute_patch_size(database)
            if self.config['patch_size'] is None:
                self.config['patch_size'] = initial_current_size
            else:
                if initial_current_size > self.config['patch_size']:
                    logging.error('BaseNetStackRoom: Cannot import the database because it exceeds the patch size.')
                    return False

            if database.mapping[1]:
                if not self.config['database_information']['is_categorical']:
                    self.config['database_information'] = INITIALIZE_CATEGORICAL
                self.config['database_information']['label_map'] = database.mapping[0]
            else:
                if self.config['database_information']['is_categorical']:
                    self.config['database_information'] = INITIALIZE_NON_CATEGORICAL
                self.config['database_information']['fuzzy_map'] = {'ranges': database.mapping[0][0],
                                                                    'labels': database.mapping[0][1]}
        else:
            size_cond = self.config['patch_size'] >= self.__compute_patch_size(database)
            type_cond = self.config['database_data_type'] == database.dtype
            shape_cond = self.config['database_shape'] == database.shape
            cat_cond = self.config['database_information']['is_categorical'] == database.mapping[1]
            if self.config['database_information']['is_categorical']:
                map_cond = self.config['database_information']['label_map'] == database.mapping[0]
            else:
                map_cond = self.config['database_information']['fuzzy_map'] == {'ranges': database.mapping[0][0],
                                                                                'labels': database.mapping[0][1]}
            if not size_cond:
                logging.error('BaseNetStackRoom: Cannot import the database because it exceeds the patch size.')
                return False
            if not type_cond or not shape_cond:
                logging.error('BaseNetStackRoom: Cannot import the database because it has different data type.')
                return False
            if not cat_cond or not map_cond:
                logging.error('BaseNetStackRoom: Cannot import the database because it has different labels.')
                return False

        # Compute class balance or statistical balance:
        database.batch_size = self.config['batch_size']
        database.name = f'{self.config["number_of_databases"]}_{database.name}'
        balance = {'name': database.name, 'train': 100 * np.mean(database.ytrain, axis=0),
                   'val': 100 * np.mean(database.yval, axis=0), 'test': 100 * np.mean(database.ytest, axis=0),
                   'size': database.size}
        if database.mapping[1]:
            self.config['database_information']['class_count'].append(balance)
            _bal = {'train': np.mean(np.array([_['train'] for _ in self.config['database_information']['class_count']]),
                                     axis=0),
                    'val': np.mean(np.array([_['val'] for _ in self.config['database_information']['class_count']]),
                                   axis=0),
                    'test': np.mean(np.array([_['test'] for _ in self.config['database_information']['class_count']]),
                                    axis=0)}
            self.config['database_information']['class_balance'] = _bal
        else:
            self.config['database_information']['statistical_count'].append(balance)

            _ba = {'train': np.mean(np.array([_['train'] for _ in
                                              self.config['database_information']['statistical_count']]), axis=0),
                   'val': np.mean(np.array([_['val'] for _ in
                                            self.config['database_information']['statistical_count']]), axis=0),
                   'test': np.mean(np.array([_['test'] for _ in
                                             self.config['database_information']['statistical_count']]), axis=0)}
            self.config['database_information']['statistical_count'] = _ba
        self.config['number_of_databases'] += 1

        # Save the database and compute the size:
        database.save(f'{self.room_path}/{__storage_dir__}/{database.name}.db')
        current_size = os.path.getsize(f'{self.room_path}/{__storage_dir__}/{database.name}.db') / 1_000_000
        self.config['current_sizes_mb'][database.name] = current_size
        self.config['current_size_mb'] += current_size
        self.__is_built = True
        return True

    @staticmethod
    def __compute_patch_size(database: BaseNetDatabase) -> float:
        database.save('./.tmp.db')
        size = os.path.getsize('./.tmp.db') / 1_000_000
        os.remove('./.tmp.db')
        return size

    def __repr__(self):
        _header = f'\n|===========================================|\n' \
                  f'| <BaseNetStackRoom (BNSR) with {self.config["number_of_databases"]:3d} shards> |\n' \
                  f'|===========================================|\n' \
                  f'|\t\t\t\t\t\t\t\t\t\t\t|\n'
        _info = f"|\t> Patch size:\t{self.config['patch_size']:5d} MB\t\t\t\t|\n" \
                f"|\t> Batch size:\t{self.config['batch_size']:5d} samples\t\t\t|\n" \
                f"|\t> Store size:\t{round(self.config['current_size_mb']):5d} MB\t\t\t\t|\n" \
                f"|\t> Categorical:\t\t{self.config['database_information']['is_categorical']}\t\t\t\t|\n" \
                f'|\t\t\t\t\t\t\t\t\t\t\t|\n'
        _end = f'|===========================================|\n'
        return f"{_header}{_info}{_end}"
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
