# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
import ray
import os


# -----------------------------------------------------------
class RayScope:
    def __init__(self, cluster_ip_address: str = None, ray_runtime_env: dict = None, run_here=True):

        if run_here:
            try:
                ray.init(runtime_env=ray_runtime_env)
                self.indoor_context = True
            except Exception as ex:
                logging.error(f'RayCluster: Could not initialize a local instance: {ex}.')
                self.indoor_context = None
        else:
            self.indoor_context = None

        if cluster_ip_address is not None:
            try:
                self.outdoor_context = ray.init(address=f'ray://{cluster_ip_address}:10001', allow_multiple=True)
            except Exception as ex:
                logging.error(f'RayCluster: Could not connect to {cluster_ip_address}: {ex}.')
                self.outdoor_context = None
        else:
            self.outdoor_context = None

        self.bind_function_in = None
        self.bind_function_ou = None
        self.function = None
        self.__total_cores = os.cpu_count()
        self.indoor_cores = self.__total_cores - 2
        if self.outdoor_context is not None:
            with self.outdoor_context:
                self.outdoor_cores = round(ray.available_resources()['CPU'])
        else:
            self.outdoor_cores = 0
        self.indoor_results = list()
        self.outdoor_results = list()
        self.scope_information = {'ip': cluster_ip_address,
                                  'c_cpus': self.outdoor_cores,
                                  'cpus': self.indoor_cores,
                                  'seg': round(self.indoor_cores / (self.indoor_cores + self.outdoor_cores), 3)}

    def bind(self, function):
        self.bind_function_in = ray.remote(num_cpus=self.indoor_cores)(function)
        if self.outdoor_context is not None:
            self.bind_function_ou = ray.remote(num_cpus=self.outdoor_cores)(function)
        self.function = function
        return self

    def bind_actor(self, actor, method_name):
        ray_actor_in = ray.remote(num_cpus=self.indoor_cores)(actor)
        self.bind_function_in = getattr(ray_actor_in, method_name)
        if self.outdoor_context is not None:
            ray_actor_ou = ray.remote(num_cpus=self.outdoor_cores)(actor)
            self.bind_function_ou = getattr(ray_actor_ou, method_name)
        self.function = getattr(actor, method_name)

    def run(self, arg_list: list[tuple], kwarg_list: list[dict] = None):
        if kwarg_list is None:
            kwarg_list = [{} for _ in arg_list]

        if self.outdoor_context is not None:
            if self.indoor_context is not None:
                partial_cut = self.indoor_cores
            else:
                partial_cut = 0
            arg_list_pipe = arg_list[:partial_cut]
            kwarg_list_pipe = kwarg_list[:partial_cut]
            arg_list_ray = arg_list[partial_cut:]
            kwarg_list_ray = kwarg_list[partial_cut:]

            if self.indoor_context is not None:
                self.indoor_results = [self.bind_function_in.remote(*arg, **kwarg)
                                       for arg, kwarg in zip(arg_list_pipe, kwarg_list_pipe)]
            with self.outdoor_context:
                self.outdoor_results = [self.bind_function_ou.remote(*arg, **kwarg)
                                        for arg, kwarg in zip(arg_list_ray, kwarg_list_ray)]
        else:
            if self.indoor_context is not None:
                self.indoor_results = [self.bind_function_in.remote(*arg, **kwarg)
                                       for arg, kwarg in zip(arg_list, kwarg_list)]
                self.outdoor_results = list()
            else:
                self.indoor_results = [self.function(*arg, **kwarg) for arg, kwarg in zip(arg_list, kwarg_list)]
        return self

    def get(self):
        ret_vals = list()

        if self.indoor_context is not None:
            for result in self.indoor_results:
                ret_vals.append(ray.get(result))

        if self.outdoor_context is not None:
            with self.outdoor_context:
                for result in self.outdoor_results:
                    ret_vals.append(ray.get(result))

        if self.indoor_context is None and self.outdoor_context is None:
            ret_vals = self.indoor_results

        return ret_vals

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.indoor_context is not None and self.outdoor_context is not None:
            try:
                ray.shutdown()
                assert ray.is_initialized()
            except AssertionError:
                ray.shutdown()
            except Exception as ex:
                raise ex
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
