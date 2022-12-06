# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
from .orchestrator import ComputationalOrchestrator
from .ray_cluster import RayCluster
from .pipeline import ComputationalPipeline


# -----------------------------------------------------------
class ComputationalScope:
    def __init__(self, cluster_ip_address: str = None, ray_runtime_env: dict = None):
        self.pipeline = ComputationalPipeline()
        if cluster_ip_address:
            self.cluster = RayCluster(cluster_ip_address, ray_runtime_env)
            self.orchestrator = ComputationalOrchestrator(has_cluster=True)
        else:
            self.cluster = None
            self.orchestrator = ComputationalOrchestrator(has_cluster=False)

    def bind(self, function):
        self.pipeline.bind(function)
        if self.cluster:
            self.cluster.bind(function)
        return self

    def run(self, arg_list: list[tuple], kwarg_list: list[dict] = None):
        if kwarg_list is None:
            kwarg_list = [{} for _ in arg_list]
        if self.cluster:
            partial_cut = round(self.orchestrator.partial * len(arg_list))
            arg_list_pipe = arg_list[:partial_cut]
            kwarg_list_pipe = kwarg_list[:partial_cut]
            arg_list_ray = arg_list[partial_cut:]
            kwarg_list_ray = kwarg_list[partial_cut:]
            self.pipeline.run(arg_list_pipe, kwarg_list_pipe)
            self.cluster.run(arg_list_ray, kwarg_list_ray)
        else:
            self.pipeline.run(arg_list, kwarg_list)
        return self

    def get(self):
        ret_vals = self.pipeline.get()
        if self.cluster:
            ret_vals.extend(self.cluster.get())
        return ret_vals

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cluster:
            self.cluster.close()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
