# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import multiprocessing
import time
import numpy as np
import ray
import joblib
from ray.util.multiprocessing import Pool as rPool


# -----------------------------------------------------------
def myfunc(order, retval=None):
    a = np.random.random((order, order))
    b = np.random.random((order, order))
    _retval = np.multiply(a, b)
    if retval is not None:
        retval.put(_retval)
    return _retval

# def myfunc(order, retval=None):
#     ask = 0
#     for i in range(order * 1_000):
#         ask += i ** 2
#     return ask


if __name__ == '__main__':

    print('[+] Connected...')
    timestamps = list()
    args = tuple([11_500 for _ in range(7)])

    print('\t[+] Single core...')
    tik = time.perf_counter()
    results = [myfunc(arg) for arg in args]
    tak = time.perf_counter() - tik
    print(tak)
    timestamps.append(round(tak, 4))

    print('\t[+] Pool...')
    tik = time.perf_counter()
    pool = multiprocessing.Pool(processes=len(args))
    results = pool.map(myfunc, args)
    tak = time.perf_counter() - tik
    print(tak)
    timestamps.append(round(tak, 4))

    print('\t[+] Process...')
    tik = time.perf_counter()
    _results = list()
    results = list()
    for arg in args:
        this = multiprocessing.Queue()
        _results.append(this)
        p = multiprocessing.Process(target=myfunc, args=(arg, this))
        p.start()
    for q in _results:
        results.append(q.get())
    tak = time.perf_counter() - tik
    print(tak)
    timestamps.append(round(tak, 4))

    print('\t[+] Ray Pool...')
    _tik = time.perf_counter()
    ray.init()
    tik = time.perf_counter()
    init_time = round(tik - _tik, 4)
    pool = rPool(processes=len(args))
    results = pool.map(myfunc, args)
    tak = time.perf_counter() - tik
    ray.shutdown()
    close_time = round(0, 4)
    print(tak)
    timestamps.append(round(tak, 5))

    print('\t[+] Joblib...')
    tik = time.perf_counter()
    results = joblib.Parallel(n_jobs=len(args))(joblib.delayed(myfunc)(arg) for arg in args)
    tak = time.perf_counter() - tik
    print(tak)
    timestamps.append(round(tak, 4))

    _text_ = f'\n\t\t\t\tTimestamp\t\tInit\t\tClosure\t\tTotal\n\n' \
             f'Single core:\t{timestamps[0]}\t\t\t-\t\t\t-\t\t\t{timestamps[0]}\n' \
             f'MP Pool:\t\t{timestamps[1]}\t\t\t-\t\t\t-\t\t\t{timestamps[1]}\n' \
             f'Multiprocess:\t{timestamps[2]}\t\t\t-\t\t\t-\t\t\t{timestamps[2]}\n' \
             f'Joblib:\t\t\t{timestamps[4]}\t\t\t-\t\t\t-\t\t\t{timestamps[4]}\n' \
             f'RayPool:\t\t{timestamps[3]}\t\t\t{init_time}\t\t{close_time}\t\t\t{timestamps[3] + init_time + close_time}\n'
    print(_text_)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
