# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
import os
import sys
from test_wrapper import basenet_test
from messages import TopLevelMessages, ErrorMessages
from basenet import __version__

from deeplearning import basenet_database_test, basenet_compiler_test, basenet_model_test, basenet_feeder_test

LOG_PATH = './testing.log'
LOG_FORMAT = '[%(asctime)s:{%(funcName)14s:%(lineno)4d}:%(levelname)s] - %(message)s'
DAY_FORMAT = '%m-%d %H:%M:%S'

REQUIREMENTS_PATH = r'../requirements.txt'
SPECIAL_IMPORT_MAPPING = {'pyyaml': 'yaml', 'pillow': 'PIL', 'logging': 'sys'}
# -----------------------------------------------------------


def set_up_logging():
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
    logging.basicConfig(level=logging.DEBUG, filename=LOG_PATH, filemode='w', format=LOG_FORMAT, datefmt=DAY_FORMAT)
    root_logger = logging.getLogger()
    log_formatter = logging.Formatter(LOG_FORMAT)
    file_handler = logging.FileHandler(LOG_PATH)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    return root_logger


def check_requirements(logger) -> bool:
    # Get the requirements and version of the API.
    reqs = list()
    version = list()
    with open(REQUIREMENTS_PATH, 'r', encoding='utf-8') as file:
        for line in file:
            if line[0] != '#':
                if '>' in line:
                    reqs.append(line.split('>')[0].lower())
                    version.append(line.split('>')[-1])
                elif '<' in line:
                    reqs.append(line.split('<')[0].lower())
                    version.append(line.split('<')[-1])
                elif '~' in line:
                    reqs.append(line.split('~')[0].lower())
                    version.append(line.split('~')[-1])
                elif '=' in line:
                    reqs.append(line.split('=')[0].lower())
                    version.append(line.split('=')[-1])

    # Check the imports in the scope.
    all_imports_correct = True
    for req, ver in zip(reqs, version):
        try:
            if req in SPECIAL_IMPORT_MAPPING:
                _req = SPECIAL_IMPORT_MAPPING[req]
            else:
                _req = req
            __import__(_req)
        except ImportError:
            all_imports_correct = False
            if ver:
                logger.error(f'{ErrorMessages.imports}:{req} version {ver.replace("=", "")}')
            else:
                logger.error(f'{ErrorMessages.imports}:{req}')
    return all_imports_correct


def run_tests(*args):
    # Set up logging.
    errors: (int, str) = 0
    logger = set_up_logging()
    logger.info(f'Logging initialization with framework version: {__version__}')
    logger.info(TopLevelMessages.connected)

    if '--install' in args or '--all' in args:
        # Testing.
        if check_requirements(logger):
            logger.info(f'\tAll the requirements are installed correctly.')
        else:
            logger.warning(f'\tSome of the requirements are not installed or not correctly installed, '
                           f'please check the requirements.txt file, some are missing.')
            args = ()
            errors = 'INSTALLATION'

    if '--deeplearning' in args or '--all' in args:
        # DeepLearningTest.
        logger.info(TopLevelMessages.deeplearning_start)
        errors += 0 if basenet_test(logger, preamble=TopLevelMessages.dl_inner, test=basenet_database_test,
                                    timeout=3, timer=5) else 1
        errors += 0 if basenet_test(logger, preamble=TopLevelMessages.dl_inner, test=basenet_compiler_test,
                                    timeout=3, timer=5) else 1
        errors += 0 if basenet_test(logger, preamble=TopLevelMessages.dl_inner, test=basenet_model_test,
                                    timeout=5, timer=30) else 1
        errors += 0 if basenet_test(logger, preamble=TopLevelMessages.dl_inner, test=basenet_feeder_test,
                                    timeout=5, timer=10) else 1
        logger.info(TopLevelMessages.deeplearning_finish)

    if '--metaheuristic' in args or '--all' in args:
        # MetaHeuristic
        logger.info(TopLevelMessages.metaheuristic_start)
        # errors += 0 if basenet_test(logger, preamble=TopLevelMessages.dl_inner, test=basenet_database_test,
        #                             timeout=3, timer=5) else 1
        logger.info(TopLevelMessages.metaheuristic_finish)

    logger.info(f'{TopLevelMessages.error_report}{errors}{TopLevelMessages.error_report_end}')
    logger.info(TopLevelMessages.disconnected)


if __name__ == '__main__':
    _args = list()
    for arg in sys.argv:
        if '--' in arg:
            _args.append(arg)
    run_tests(*_args)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
