# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import dataclasses
# -----------------------------------------------------------
_ERRORS = \
    {
        'ARE0': 'Expected [', 'ARE1': '] output, got [', 'ARE2': ']',  # Assertion error resolve.

        # Generic errors:
        'E000': 'An error occurred while testing the module (E000)',  # Generic failure.
        'E001': 'An error occurred while testing the module (E001)',  # Assertion error.
        'E002': 'An error occurred while testing the module (E002)',  # Value error.
        'E003': 'An error occurred while testing the module (E003)',  # Runtime error.
        'E004': 'A timeout error occurred while testing the module (E004)',  # Performance warning.
        'E005': 'A timeout error occurred while testing the module (E005)',  # Timeout error.
        'E006': 'Error while trying to import a requirement module (E006)',  # Import error.
        'E007': 'Error in the test watchdog (E007)',  # Watchdog error.

        # DeepLearning errors:
        'E010': 'Error in the BaseNetDatabase, representation (E010)',
        'E011': 'Error in the BaseNetDatabase, add and divide operations (E011)',
    }


def assertion_to_message(ex):
    args = ex.args[0]
    numb = args[0]
    expected = args[1]
    got = args[2]
    return _ERRORS[f'E{numb:03d}'], expected, got


def do_assert(a, b, n_error):
    assert a == b, (n_error, a, b)


@dataclasses.dataclass
class InitMessages:
    starting_test: str = 'Starting test'
    ending_test_correct: str = 'Finishing test without errors'
    ending_test_wrong: str = 'Finishing test with some errors'


@dataclasses.dataclass
class ErrorMessages:
    # Assertion error resolve:
    are0: str = _ERRORS['ARE0']
    are1: str = _ERRORS['ARE1']
    are2: str = _ERRORS['ARE2']
    # Errors:
    failed: str = _ERRORS['E000']
    assertion: str = _ERRORS['E001']
    value: str = _ERRORS['E002']
    runtime: str = _ERRORS['E003']
    timeout: str = _ERRORS['E004']
    watchdog: str = _ERRORS['E005']
    imports: str = _ERRORS['E006']
    in_watchdog: str = _ERRORS['E007']



@dataclasses.dataclass
class TopLevelMessages:
    connected: str = '[+] Connected to the BaseNetTest module'
    disconnected: str = '[-] Disconnected to the BaseNetTest module'
    imports: str = '\tChecking Python requirements'
    # Deep Learning
    dl_inner: str = '\t\tDeepLearningAPI'
    deeplearning_start: str = '\tStarting the DeepLearning API test'
    deeplearning_finish: str = '\tFinishing the DeepLearning API test'
    # MetaHeuristic
    mh_inner: str = '\t\tMetaHeuristicAPI'
    metaheuristic_start: str = '\tStarting the MetaHeuristic API test'
    metaheuristic_finish: str = '\tFinishing the MetaHeuristic API test'
    # Error report
    error_report: str = '\tTHE TEST FINISHED WITH '
    error_report_end: str = ' ERRORS'


@dataclasses.dataclass
class LowLevelError:
    # Deep Learning
    some_error: str = _ERRORS['E010']
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
