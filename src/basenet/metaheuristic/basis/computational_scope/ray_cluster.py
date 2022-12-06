# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
import ray


# -----------------------------------------------------------
class RayCluster:
    def __init__(self, ip_address, runtime_env=None):
        """
        Initializes a Ray Cluster.
        """
        self.ip = ip_address
        if runtime_env is not None:
            self.runtime_env = runtime_env
        else:
            self.runtime_env = {}
        self.base_target = None
        self.futures = []
        self.open()

    def open(self):
        """
        Creates a connection to the cluster.
        :return: The object of the Ray Cluster Class.
        """
        if not ray.is_initialized():
            ray.init(address=f'ray://{self.ip}:10001', runtime_env=self.runtime_env)
            logging.info(f'[+] Connected to {self.ip}:10001.')
        else:
            logging.info(f'[+] Already connected to {self.ip}:10001.')
        return self

    def close(self):
        """
        Closes the connection to the cluster.
        :return: None
        """
        if ray.is_initialized():
            ray.shutdown()
            logging.info(f'[-] Disconnected from {self.ip}:10001.')
        else:
            logging.info(f'[-] The cluster is already disconnected.')

    def bind(self, function):
        """
        Binds a function to the Ray Cluster.
        :param function: The function to be run in parallel in the Ray Cluster.
        :return: The object of the Ray Cluster Class.
        """
        self.base_target = ray.remote(function)
        return self

    def run(self, arg_list: list[tuple], kwarg_list: list[dict]):
        """
        Runs the bind function.
        :param arg_list: List of arguments.
        :param kwarg_list: List of kwarguments.
        :return: The object of the Ray Cluster Class.
        """
        f = []
        for args, kwargs in zip(arg_list, kwarg_list):
            f.append(self.base_target.remote(*args, **kwargs))
        self.futures = f
        return self

    def get(self):
        """
        Waits the process to be finished and recovers the return values from the bind function.
        :return: A list with the returned values.
        """
        return ray.get(self.futures)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
