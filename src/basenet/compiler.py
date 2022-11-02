# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
"""
The compiler.py file contains the BaseNetCompiler class.

Available layers:

    *   KERAS_LIST_LAYERS
    *   PREBUILT_LAYERS

Available loss functions:

    *   KERAS_LOSSES
    *   PREBUILT_LOSSES

Available optimizers:

    *   KERAS_OPTIMIZERS
    *   PREBUILT_OPTIMIZERS
"""
# Import statements:
import copy
import psutil
import os
import logging
import pickle
import yaml
from ._names import KERAS_LOSSES, KERAS_LIST_LAYERS, PREBUILT_LOSSES, PREBUILT_LAYERS, \
    KERAS_OPTIMIZERS, PREBUILT_OPTIMIZERS
from .model import BaseNetModel
from tensorflow.python.client import device_lib
from pynvml.smi import nvidia_smi
from .__special__ import __base_compiler__


# -----------------------------------------------------------
class BaseNetCompiler:
    """
    The BaseNetCompiler is a custom compiler that takes the information about the network and compiles it with the given
    parameters.

    The BaseNetCompiler also allows the user to use a .yaml file to build the network with the following format:

        compiler:
          name: <name of the model>
          input_shape:
            - <input shape of the model (I)>
            - <input shape of the model (II)>
            - <...>
          output_shape: <output shape of the model>

          compile_options:
            loss: <tf.keras loss function name>
            optimizer: <tf.keras optimizer name>
            metrics:
              - <tf.keras loss function name provided as a loss function>
              - <'accuracy' is always a good metric to analyze>

          devices:
            - <your device type>:
                name: <the name of your device in BaseNetCompiler.show_devs()>
                state: <'Idle' for nothing, 'Train' for training>

            <some device examples:>
            - cpu:
                name: "/device:CPU:0"
                state: "Idle"
            - gpu:
                name: "/device:GPU:0"
                state: "Train"
            - gpu:
                name: "/device:GPU:1"
                state: "Idle"
            - gpu:
                name: "/device:GPU:2"
                state: "Idle"

          layers:
            - layer:
                name: <layer name in tf.keras.layers>
                shape:
                    - <layer shape (I)>
                    - <layer shape (II)>
                    - <...>
                options:
                    - option:
                        name: <the name of the option in tf.keras.layers.<your layer name> or
                               "{open}/{close}_pipeline">
                        value: <the value of the option in tf.keras.layers.<your layer name>>

            <some layer examples:>
            - layer:
                name: "Flatten"
                shape:
                options:

            - layer:
                name: "Dense"
                shape:
                  - 128
                options:
                  - option:
                      name: "activation"
                      value: "relu"

            - layer:
                name: "Dense"
                shape:
                  - 64
                options:

            - layer:
                name: "Dense"
                shape:
                  - 32
                options:
                  - option:
                      name: "activation"
                      value: "sigmoid"

            - layer:
                name: "open_pipeline"
                shape:
                options:

            - layer:
                name: "Dense"
                shape:
                  - 32
                options:
                  - option:
                      name: "activation"
                      value: "sigmoid"

            - layer:
                name: "open_pipeline"
                shape:
                options:

            - layer:
                name: "Dense"
                shape:
                  - 32
                options:
                  - option:
                      name: "activation"
                      value: "sigmoid"

            - layer:
                name: "close_pipeline"
                shape:
                options:

    When open_pipeline is provided, the model creates a separate pipeline for the incoming layers. If more than one
    open_pipeline is provided, more pipelines will be added. When close_pipeline is provided, a
    tf.keras.layers.Concatenate layer is added into the model to close all the previous models into the main pipeline.

    This compiler implements some TensorFlow functions to list the GPU devices.
    """
    def __init__(self, io_shape: tuple, compile_options: dict, devices: dict, layers: list[dict] = None,
                 name: str = 'current_model', verbose: bool = False):
        """
        Build the BaseNetCompiler class.
        :param io_shape: Input-output shape [(input,), output].
        :param compile_options: Dictionary of compiling options {loss: , optimizer: , metrics: }.
        :param devices: {device: role}. Consider calling: BaseNetCompiler.show_devs().
        :param layers: List of layers: {name: ( (shape,) , {'args': args} )}.
        :param name: Name of the model.
        :param verbose: Print state and errors in the BaseNetCompiler.
        """
        if layers is not None:
            self.layers: list[dict] = layers
        else:
            self.layers: list[dict] = []
        self.compile_options: dict = compile_options
        self.devices: dict = devices
        self.io_shape: tuple = io_shape
        self.name: str = name

        self._verbose = verbose
        self.is_compiled = False
        self.is_valid: bool = False

        self._check()

    @staticmethod
    def build_from_yaml(path: str = __base_compiler__, verbose: bool = False):
        """
        This function builds the BaseNetCompiler from a formatted .yaml file.
        :param path: Path of the .yaml file with the compiler directives.
        :param verbose: Enables print debugging.
        :return: The compiler object of the class BaseNetCompiler.
        """
        with open(path, 'r', encoding='utf-8') as file:
            compiler = yaml.load(file, yaml.FullLoader)['compiler']
        name = compiler['name']
        io_shape = (tuple(compiler['input_shape']), compiler['output_shape'])
        options = compiler['compile_options']

        devices = {}
        for _dev in compiler['devices']:
            for typ, dev in _dev.items():
                if typ == 'cpu' or typ == 'gpu':
                    devices[dev['name']] = dev['state']

        layers = []
        for __layer in compiler['layers']:
            for typ, layer in __layer.items():
                if typ == 'layer':
                    _layer = layer
                    if layer['shape'] is not None:
                        _layer['shape'] = tuple(layer['shape'])
                    else:
                        _layer['shape'] = (None,)
                    _options = {}
                    if layer['options'] is not None:
                        for _option_ in layer['options']:
                            for key, option in _option_.items():
                                if key == 'option':
                                    _options[option['name']] = option['value']
                    _layer['options'] = _options

                    layers.append({_layer['name']: (_layer['shape'], _layer['options'])})

        self = BaseNetCompiler(io_shape=io_shape, compile_options=options, devices=devices, layers=layers, name=name,
                               verbose=verbose)
        return self

    def compile(self, name: str = None):
        """
        This method of the BaseNetCompiler generates a BaseNetModel from a valid BaseNetCompiler.
        :param name: Name of the model. This variable overrides the name parameter of the BaseNetCompiler
        if it is provided.
        :return: (BaseNetCompiler, BaseNetModel)
        """
        if self.is_valid:
            try:
                if name is not None:
                    model = BaseNetModel(self, name=name, verbose=self._verbose)
                else:
                    model = BaseNetModel(self, name=self.name, verbose=self._verbose)
                self.is_compiled = True
                return model
            except Exception as ex:
                logging.error(f'BaseNetCompiler: An exception occurred while building the model: {ex}')
                return None
        else:
            if self._verbose:
                logging.warning('BaseNetCompiler: The model is not valid for compiling.')
            return None

    @staticmethod
    def show_devs():
        """
        This method lists the devices in the current machine with the BaseNetModel dictionary format.
        :return: A dictionary with all the available devices in the machine.
        """
        _retval_ = dict()
        libdiv = device_lib.list_local_devices()
        for dev in libdiv:
            _retval_[dev.name] = 'Idle'
        _retval_[libdiv[0].name] = 'Train'
        return _retval_

    def pop(self, index: int) -> dict:
        """
        This method pops out a layer from the architecture.
        :param index: The place of the layer in the layers list.
        :return: The popped layer.
        """
        popped = self.layers.pop(-index)
        return popped

    def add(self, layer: dict, where: int = -1):
        """
        This function adds a new layer on the bottom of the architecture.
        :param layer: The layer to be added to the compiler.
        :param where: The place to be inserted in the architecture.
        :return: Bypasses the current BaseNetCompiler.
        """
        if where == -1:
            self.layers.append(layer)
        else:
            self.layers.insert(where, layer)
        if not self._check(from_which=where):
            self.layers.pop(-1)
        return self

    # Save and load methods:
    def save(self, _compiler_path: str) -> bool:
        """
        This function saves the BaseNetCompiler in a .cpl format.
        :param _compiler_path: Path where the BaseNetCompiler is being saved in the file system.
        :return: True if the saving was successful. False if not.
        """
        if _compiler_path:
            if '.cpl' not in _compiler_path:
                __compiler_path = f'{_compiler_path}.cpl'
            else:
                __compiler_path = _compiler_path
            with open(__compiler_path, 'wb') as file:
                pickle.dump(self, file)
            return True
        else:
            return False

    def export(self, _compiler_path: str) -> bool:
        """
        This function export the BaseNetCompiler to a .yaml format.
        :param _compiler_path: Path where the BaseNetCompiler .yaml file is being saved in the file system.
        :return: True if the saving was successful. False if not.
        """
        _devices_ = list()
        for name, role in self.devices.items():
            if 'cpu' in name or 'CPU' in name:
                _devices_.append({'cpu': {'name': name, 'state': role}})
            if 'gpu' in name or 'GPU' in name:
                _devices_.append({'gpu': {'name': name, 'state': role}})

        _layers_ = list()
        for layer in self.layers:
            _options_ = list()
            for key, item in layer.items():
                _options_.append({'option': {'name': key, 'value': item}})
            _layers_.append({'layer': {'name': layer['name'], 'shape': layer['shape'], 'options': _options_}})

        _yaml_full_ = {
            'name': self.name,
            'input_shape': list(self.io_shape[0]),
            'output_shape': self.io_shape[1],
            'compile_options': self.compile_options,
            'devices': _devices_,
            'layers': _layers_
        }
        yaml_on = {'compiler': _yaml_full_}
        if _compiler_path:
            if '.yaml' not in _compiler_path:
                __compiler_path = f'{_compiler_path}.yaml'
            else:
                __compiler_path = _compiler_path
            with open(__compiler_path, 'wb') as file:
                yaml.dump(yaml_on, file, default_flow_style=False)
            return True
        else:
            return False

    @staticmethod
    def load(_compiler_path: str):
        """
        This function loads a BaseNetCompiler from a .cpl file format.
        :param _compiler_path: Path where the BaseNetCompiler is being loaded from the file system.
        :return: The BaseNetCompiler if the saving was successful. 'None' if not.
        """
        if _compiler_path:
            if '.cpl' not in _compiler_path:
                __compiler_path = f'{_compiler_path}.cpl'
            else:
                __compiler_path = _compiler_path
            with open(__compiler_path, 'rb') as file:
                compiler = pickle.load(file)
            return compiler
        else:
            return None

    @staticmethod
    def set_up_devices(let_free_ram: float = 0.8):
        """
        This function automatically sets the available devices for use in the models.
        Note that if your free VRAM > free RAM the TF framework will report OUT_OF_MEMORY errors.
        This function disables some GPUs from the Python scope.
        :param let_free_ram: The percentage of RAM not to be used.
        :return: Nothing, this function just sets up an internal API config file.
        """
        try:
            # Obtain the usable devices without OUT_OF_MEMORY errors.
            nvsmi = nvidia_smi.getInstance()
            query_devs = nvsmi.DeviceQuery('memory.free, name')['gpu']
            devs = []
            total_vram = 0
            _total_ram = psutil.virtual_memory().free / 1000000
            total_usable_vram = _total_ram * let_free_ram
            for dev in query_devs:
                total_vram += dev['fb_memory_usage']['free']
                devs.append({dev['product_name']: dev['fb_memory_usage']['free']})
            sorted_devs = copy.copy(devs)
            sorted_devs.sort(key=lambda x: x.items())
            free_ram = total_usable_vram
            usable_devices = []

            for dev in sorted_devs:
                k_i = dev.items()
                for name, dev_vram in k_i:
                    if free_ram > dev_vram:
                        usable_devices.append(name)
                        free_ram -= dev_vram

            _visible_config_ = ''
            for index, dev in enumerate(devs):
                for name in dev.keys():
                    if name in usable_devices:
                        if _visible_config_:
                            _visible_config_ = f'{_visible_config_},{index}'
                        else:
                            _visible_config_ = f'{index}'

            # Set up the config in the os.environ variable and the API config file.
            # with open(__config_path__, 'w', encoding='utf-8') as file:
            #     cfg = json.load(file)
            #     cfg['gpu_devices'] = _visible_config_
            os.environ["CUDA_VISIBLE_DEVICES"] = _visible_config_
        except Exception as ex:
            logging.error(f'NVML: Nvidia drivers are not detected: {ex}')
            logging.warning('Configured 0 GPUs for the session, NVIDIA Drivers are not detected.')
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Checking:
    def _check(self, from_which=0):
        if self._check_layers(from_which=from_which) and self._check_devices() and self._check_compiler_options():
            self.is_valid = True
            return True
        else:
            return False

    def _check_compiler_options(self):
        if 'loss' not in self.compile_options or 'optimizer' not in self.compile_options:
            if self._verbose:
                logging.error(f'BaseNetCompiler: The compiler options "loss" and "optimizer" are mandatory.')
            self.is_valid = False
            return False
        else:
            lossfunc = self.compile_options['loss']
            optifunc = self.compile_options['optimizer']
            if lossfunc not in PREBUILT_LOSSES and lossfunc not in KERAS_LOSSES:
                if self._verbose:
                    logging.error(f'BaseNetCompiler: The loss function {lossfunc} is not in the API.')
                self.is_valid = False
                return False
            if optifunc not in KERAS_OPTIMIZERS and optifunc not in PREBUILT_OPTIMIZERS:
                if self._verbose:
                    logging.error(f'BaseNetCompiler: The optimizer {optifunc} is not in the API.')
                self.is_valid = False
                return False
            return True

    def _check_devices(self):
        current_devs = self.show_devs()
        for dev in self.devices:
            if dev not in current_devs:
                if self.devices[dev] != 'Idle':
                    if self._verbose:
                        logging.warning(f'BaseNetCompiler: The device {dev} is not available in this machine '
                                        f'right now.')
                    self.is_valid = False
                    return False
        return True

    def _check_layers(self, from_which=0):
        # Formatting:
        #
        #   {'layer_name': ( (shape0, shape1, ...) , {'arg1': arg1, 'arg2': arg2} )}
        #
        for layer in self.layers[from_which:]:
            for key, item in layer.items():
                if key not in KERAS_LIST_LAYERS and key not in PREBUILT_LAYERS:
                    self.is_valid = False
                    if self._verbose:
                        logging.warning(f'BaseNetCompiler: layer {key} not found.')
                    return self.is_valid
                else:
                    if not isinstance(item, tuple):
                        self.is_valid = False
                        if self._verbose:
                            logging.warning(f'BaseNetCompiler: layer {key} does not contain a tuple.')
                        return self.is_valid
                    else:
                        if isinstance(item[0], tuple):
                            if isinstance(item[1], dict):
                                return True
                            else:
                                self.is_valid = False
                                if self._verbose:
                                    logging.warning(
                                        f'BaseNetCompiler: layer {key}: {item[1]} does not contain a dict.')
                                return self.is_valid
                        else:
                            self.is_valid = False
                            if self._verbose:
                                logging.warning(f'BaseNetCompiler: layer {key}: {item[0]} does not contain a '
                                                f'tuple: (shapes,).')
                            return self.is_valid

    # Build methods:
    def __repr__(self):
        return f'Compiler with {len(self.layers)} layers, options:\n{self.compile_options}'

    def __add__(self, other):
        self.layers.append(other)
        if not self._check(from_which=-1):
            self.layers.pop(-1)

    def __bool__(self):
        return self.is_valid
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
