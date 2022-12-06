# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import os
import ray


# -----------------------------------------------------------
class ComputationalOrchestrator:
    def __init__(self, has_cluster: bool = False):
        """
        Creates a Computational Orchestrator instance.
        :param has_cluster: Tells the oechestrator if we want to use a cluster.
        """
        self.cpus = os.cpu_count() - 2
        if has_cluster:
            self.ray_cpus = ray.available_resources()['CPU'] - 2
        else:
            self.ray_cpus = 0
        self.partial = self.cpus / (self.ray_cpus + self.cpus)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
