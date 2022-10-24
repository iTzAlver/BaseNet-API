# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import pickle
from ._model import Model
from tensorflow.python.client import device_lib


# -----------------------------------------------------------
class Compiler:
    def __init__(self, io_shape: tuple, layers: list, shapes: list[tuple], kwds: list[list], args: list[list],
                 compiler: dict, devices: dict, verbose: bool = True):
        """
        Build the compiler class.
        :param io_shape: Input-output shape [(input,), output].
        :param layers: List of layers.
        :param shapes: List of shapes [(x,), (y, z), ...].
        :param kwds: Keywords: [['kw00', kw01'], ['kw10'], [None], ...].
        :param args: Argument: [[arg00, arg01], [arg10], [None], ...].
        :param compiler: Dictionary of compiling options {loss: , optimizer: , metrics: }.
        :param devices: Consider calling: Compiler.devices().
        :param verbose: Print state and errors in the Compiler.
        """
        self.layers: list[str] = layers
        self.shapes: list[tuple] = shapes
        self.args: list[dict] = []
        self.compiler: dict = compiler
        self.is_valid: bool = True
        self.devices: dict = devices
        self.io_shape: tuple = io_shape

        self._verbose = verbose

        if len(kwds) != len(args) != len(layers) != len(shapes):
            if verbose:
                print('Number of keywords not equal to number of arguments.')
            self.compiled = False
        else:
            for kwd_s, arg_s in zip(kwds, args):
                dik = dict()
                if len(kwd_s) != len(arg_s):
                    if verbose:
                        print(f'Number of member of keywords not equalt to the arguments for the memeber: {kwd_s}')
                    self.compiled = False
                else:
                    for kwd, arg in zip(kwd_s, arg_s):
                        if kwd is not None:
                            dik[kwd] = arg
                self.args.append(dik)

    def compile(self, name='current_model'):
        if self.is_valid:
            return Model(self, name=name)
        else:
            if self._verbose:
                print('The model is not valid for compiling.')
            return None

    @staticmethod
    def show_devs():
        _retval_ = dict()
        libdiv = device_lib.list_local_devices()
        for dev in libdiv:
            _retval_[dev.name] = 0
        _retval_[libdiv[0].name] = 1
        return _retval_

    def save(self, _compiler_path):
        if _compiler_path:
            with open(_compiler_path, 'wb') as file:
                pickle.dump(self, file)
            return True
        else:
            return False

    @staticmethod
    def load(_compiler_path):
        if _compiler_path:
            with open(_compiler_path, 'rb') as file:
                compiler = pickle.load(file)
            return compiler
        else:
            return None

    def __repr__(self):
        return f'Compiler with {len(self.layers)} layers, options:\n{self.compiler}'
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
