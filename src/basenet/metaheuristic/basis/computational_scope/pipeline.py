# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import multiprocessing
# -----------------------------------------------------------


class ComputationalPipeline:
    def __init__(self):
        """
        Initializes a Computational Pipeline.
        """
        self.base_target = None
        self.process: list[multiprocessing.Process] = []
        self.futures: list[multiprocessing.Queue] = []

    def bind(self, function):
        """
        Binds a function in the Computational pipeline.
        :param function: The function to be run in parallel.
        :return: The object of the Computational Pipeline Class.
        """
        self.base_target = function
        return self

    def run(self, arg_list: list[tuple], kwarg_list: list[dict]):
        """
        Runs the bind function.
        :param arg_list: List of arguments.
        :param kwarg_list: List of kwarguments.
        :return: The object of the Computational Pipeline Class.
        """
        if self.base_target is not None:
            p = []
            q = []
            for args, kwargs in zip(arg_list, kwarg_list):
                q.append(multiprocessing.Queue())
                kwargs['__futures__'] = q[-1]
                kwargs['__function__'] = self.base_target
                p.append(multiprocessing.Process(target=self._target, args=args, kwargs=kwargs))
                p[-1].start()
            self.process = p
            self.futures = q
        return self

    def get(self):
        """
        Waits the process to be finished and recovers the return values from the bind function.
        :return: A list with the returned values.
        """
        for q in self.futures:
            while q.empty():
                pass
        return [q.get() for q in self.futures]

    @staticmethod
    def _target(*args, **kwargs):
        retval = kwargs['__function__'](*args, **kwargs)
        kwargs['__futures__'].put(retval)
# if __name__ == '__main__':
#     cp = ComputationalPipeline().bind(print_date).run(arg_list=[('hola',), ('pepe',), (100,)],
#                                                       kwarg_list=[{}, {}, {}])
#     print(cp.get())
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
