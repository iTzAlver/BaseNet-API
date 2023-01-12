# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import threading
import time
import ctypes
from messages import InitMessages, ErrorMessages, assertion_to_message


# -----------------------------------------------------------
def basenet_test(logger, preamble, test, timeout: int = 0, timer: int = 0) -> bool:
    """
    This is a top level wrapper for a test of the BaseNet-API.
    :param logger: The top-level logger.
    :param preamble: A string with the top level message.
    :param test: The test function.
    :param timeout: A timeout value for performance warning in seconds.
    :param timer: A timeout value for error in seconds.
    :return: True if correct, false if not.
    """
    name = f'<{test.__name__}>'
    logger.info(f'{preamble}:{InitMessages.starting_test}:{name}')
    wd = threading.Thread(target=watchdog, args=(timer, lambda: stop_condition, threading.current_thread()))
    try:
        stop_condition = False
        _starting_ = time.perf_counter()
        wd.start()
        test(logger, f'{preamble}:{name}:')
        stop_condition = True
        _elapsed_ = time.perf_counter() - _starting_
        if _elapsed_ > timeout > 0:
            logger.warning(f'{preamble}:{ErrorMessages.timeout}:{name}')
            return False
        else:
            logger.info(f'{preamble}:{InitMessages.ending_test_correct}:{name}')
            return True
    except TimeoutError:
        logger.error(f'{preamble}:{ErrorMessages.watchdog}:{name}')
        logger.info(f'{preamble}:{InitMessages.ending_test_wrong}:{name}')
    except ValueError as ex:
        logger.error(f'{preamble}:{ErrorMessages.value}:{name}:{ex}')
        logger.info(f'{preamble}:{InitMessages.ending_test_wrong}:{name}')
    except RuntimeError as ex:
        logger.error(f'{preamble}:{ErrorMessages.runtime}:{name}:{ex}')
        logger.info(f'{preamble}:{InitMessages.ending_test_wrong}:{name}')
    except AssertionError as ex:
        message, expected, got = assertion_to_message(ex)
        logger.error(f'{preamble}:{ErrorMessages.assertion}:{name}:{message}')
        logger.error(f'{preamble}:{ErrorMessages.are0}{expected}{ErrorMessages.are1}{got}{ErrorMessages.are2}')
        logger.info(f'{preamble}:{InitMessages.ending_test_wrong}:{name}')
    except Exception as ex:
        logger.error(f'{preamble}:{ErrorMessages.failed}:{name}:{ex}')
        logger.info(f'{preamble}:{InitMessages.ending_test_wrong}:{name}')
    finally:
        reraise(wd, TimeoutError)


def watchdog(timer, finished, parent):
    try:

        if timer > 0:
            init = time.time()
            while time.time() - init < timer:
                time.sleep(0.5)
            if not finished():
                reraise(parent, TimeoutError)
    except TimeoutError:
        return


# https://gist.github.com/liuw/2407154
def reraise(thread_obj, exception):
    # Thread search...
    target_tid = 0
    for tid, tobj in threading._active.items():
        if tobj is thread_obj:
            target_tid = tid
            break
    ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(target_tid, ctypes.py_object(exception))
    if ret == 0:
        raise ValueError(ErrorMessages.in_watchdog)
    elif ret > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(target_tid, 0)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
