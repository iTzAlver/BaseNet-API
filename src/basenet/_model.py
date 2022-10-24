# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import pickle
import copy
import numpy as np
from tensorflow import keras, convert_to_tensor, distribute, expand_dims
from keras.utils.vis_utils import plot_model
from ._typeoflayers import KERAS_LISTOF_TYPEOFLAYERS, PREBUILT_LAYERS, CUSTOM_LOSES
from ._logger import Logger
from ..__path_to_config__ import PATH_TO_CONFIG
from .algorithms import Subkeras
from .loss_functions import Sublosses
# -----------------------------------------------------------
import os
import json
try:
    with open(PATH_TO_CONFIG, 'r') as file_:
        cfg = json.load(file_)
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg['tensorflow']['devices_listing']
except Exception as ex:
    print('Traceback: _model.py. Path to config corrupted, try to run the script from the proper path.')
    print(f'{ex}.\nCuda did not start properly.')


# -----------------------------------------------------------
class Model:
    def __init__(self, compiler, model=None, name='current_model'):
        """
        The Class model implements an API that makes use of keras and tensorflow to build Deep Learning Models.
        :param compiler: Compiler object to build the model.
        :param model: If a keras.model is already compiled, you can import it in the model parameter, so the compiler
        will not be used.
        """
        self._bypass = False
        self.model = None
        self.compiler = compiler
        self.summary: str = 'Uncompiled model.'
        self.name = name
        if model is None:
            if self.compiler is not None:
                self.devices: dict = compiler.devices
                self.compile()
            else:
                self.devices: dict = {}
        else:
            self.model = model
            if self.compiler is not None:
                self.devices: dict = compiler.devices
            else:
                self.devices: dict = {}
            self._logtracker()

        self.history: list = []
        self.is_trained: bool = False

    def compile(self):
        if self.compiler is not None:
            self._compile()
            self._logtracker()
        else:
            print('Warning: Compiled failed because there was no compiler.')

    def _compile(self):
        compiler = self.compiler
        _scope = self._model_scope(self.devices)

        with _scope:
            # Add the input of the model.
            _inp = keras.Input(shape=compiler.io_shape[0])
            _inp._name = 'compiled-model-keras'
            _lastlay = _inp

            last_master = None
            pipeline_opened = False
            towers = []

            for layer_type, layer_shape, layer_args in zip(compiler.layers, compiler.shapes, compiler.args):
                # Core layers:
                if layer_type == 'open_pipeline':
                    if not pipeline_opened:
                        last_master = _lastlay
                    else:
                        towers.append(_lastlay)
                        _lastlay = last_master
                    pipeline_opened = True
                elif layer_type == 'close_pipeline':
                    pipeline_opened = False
                    towers.append(_lastlay)
                    _lastlay = keras.layers.concatenate(towers, **layer_args)
                    towers = []
                else:
                    if layer_type in KERAS_LISTOF_TYPEOFLAYERS and layer_type not in PREBUILT_LAYERS:
                        this_lay = getattr(keras.layers, layer_type)(*layer_shape, **layer_args)
                    elif layer_type in KERAS_LISTOF_TYPEOFLAYERS:
                        this_lay = getattr(Subkeras, layer_type)(*layer_shape, **layer_args)
                    else:
                        importmodel = keras.models.load_model(f'{layer_type}.h5')
                        importmodel._name = layer_type
                        this_lay = importmodel

                    _lastlay = this_lay(_lastlay)

            # Add the output of the model.
            out = keras.layers.Dense(compiler.io_shape[1], activation="sigmoid", name='output')(_lastlay)
            # out = keras.layers.ThresholdedReLU(0.2)(out)

            _compile = copy.copy(compiler.compiler)
            if _compile['loss'] in CUSTOM_LOSES:
                _compile['loss'] = getattr(Sublosses, compiler.compiler['loss'])

            model = keras.Model(_inp, out, name=self.name)
            model.compile(**_compile)
            self.model = model

    def model_print(self, print_path='.'):
        plot_model(self.model, to_file=f'{print_path}/compiled-model.gv.png', show_shapes=True)

    def fit(self, db, epoch=1, evaluate=True):
        self.is_trained = True
        xtrain = convert_to_tensor(np.array(db.dataset.xtrain).astype("float32") / 255)
        ytrain = convert_to_tensor(db.dataset.ytrain)
        xval = convert_to_tensor(np.array(db.dataset.xval).astype("float32") / 255)
        yval = convert_to_tensor(db.dataset.yval)
        _history = self.model.fit(xtrain, ytrain, batch_size=db.batch_size, epochs=epoch, validation_data=(xval, yval))
        history = dict()
        history['loss'] = _history.history['loss']
        if evaluate:
            history['eval'] = self._eval(xval, yval)
        self.history.append(_history.history)
        return history

    def save(self, model_path, compiler_path=''):
        self.model.save(model_path)
        if not compiler_path:
            _compiler_path = model_path.replace('.h5', '.cpl')
        elif '.cpl' in compiler_path:
            _compiler_path = compiler_path
        else:
            _compiler_path = f'{compiler_path}.cpl'
        if self.compiler:
            with open(_compiler_path, 'wb') as file:
                pickle.dump(self.compiler, file)

    @staticmethod
    def load(model_path, compiler_path=''):
        custom_obj = {}
        if not compiler_path:
            _compiler_path = model_path.replace('.h5', '.cpl')
        elif '.cpl' in compiler_path:
            _compiler_path = compiler_path
        else:
            _compiler_path = f'{compiler_path}.cpl'
        if os.path.exists(_compiler_path):
            with open(_compiler_path, 'rb') as file:
                compiler = pickle.load(file)
            custom_obj = _get_customs(compiler)
        else:
            print('Warining: The compiler path is empty, the current model has no compiler.')
            compiler = None

        model = keras.models.load_model(model_path, custom_objects=custom_obj)
        name = model_path.split('/')[-1].replace('.h5', '')
        return Model(compiler, model=model, name=name)

    @staticmethod
    def _model_scope(_devices):
        # Cretes a scope from the current devices.
        devices = []
        for _dev, role in _devices.items():
            if role == 1:
                devices.append(_dev)
        if len(devices) > 0:
            strategy = distribute.MirroredStrategy(devices=devices)
            scope = strategy.scope()
        else:
            raise ValueError('There are no training devices...')
        return scope

    def _logtracker(self):
        # Extracts the summary of the current model.
        log = Logger()
        log.start()
        self.model.summary()
        __msg = log.messages
        _msg = ''
        for msg in __msg:
            _msg = f'{_msg}{msg}'
        self.summary = _msg
        log.stop()

    @staticmethod
    def fitmodel(_model, db, queue, bypass='', epoch=1):
        model = _model
        cex = 0
        try:
            if bypass:
                model.bypass(bypass)
            # Receive the messages from master.
            msg = ''
            while msg != 'MASTER:STOP':
                try:
                    hist = model.fit(db, epoch)
                    cex = 0
                    if not queue.empty():
                        msg = queue.get()
                    queue.put(hist)
                except Exception as __ex:
                    cex += 1
                    print(f'{cex} Consecutive exceptions received from the training worker...')
                    if cex > 2:
                        print(f'Exception received from the training worker at training level: {__ex}')
                        raise __ex
        except Exception as _ex:
            print(f'Exception received from the training worker at bypassing level: {_ex}')
            queue.put('SLAVE:RETRY')
        finally:
            # Bypassing again.
            if bypass:
                model.bypass(bypass)

    def bypass(self, path='.'):
        if not self._bypass:
            self.model.save(path)
            self.model = None
            self._bypass = True
        else:
            self.model = keras.models.load_model(path, custom_objects=_get_customs(self.compiler))
            self._bypass = False
            if '.h5' not in path:
                _path = f'{path}.h5'
            else:
                _path = path
            os.remove(_path)
        return path

    def __repr__(self):
        return f'Model object with the following parameters:\nCompiler: {self.compiler}\nSummary: {self.summary}'

    def __bool__(self):
        return self.is_trained

    def __sizeof__(self):
        return len(self.compiler.layers)

    def __eq__(self, other):
        return self.compiler.layers == other.compiler.layers

    def _eval(self, xval, yval, th=0.5):
        pred = self.model.predict(xval)
        wd = Sublosses.window_diff(pred, yval, th=th)
        return wd

    def eval(self, xval, yval, th=0.5):
        try:
            _xval = convert_to_tensor(np.array(xval).astype("float32") / 255)
            _yval = convert_to_tensor(yval)
            wd = self._eval(_xval, _yval, th=th)
            return wd
        except Exception as _ex_:
            print(f'An Exception occurred while evaluation: {_ex_}')
            return 1

    def predict(self, x, expand: bool = False, th: (None, float) = None):
        _x_ = convert_to_tensor(np.array(x).astype("float32") / 255)
        if expand:
            __x = expand_dims(_x_, axis=-1)
        else:
            __x = _x_
        _y_ = self.model.predict(__x)
        if th is not None:
            __y__ = self.threshold(_y_, th)
        else:
            __y__ = _y_
        return __y__

    @staticmethod
    def threshold(y, th):
        __y__ = []
        for element in y:
            if element > th:
                __y__.append(1)
            else:
                __y__.append(0)
        return convert_to_tensor(__y__)


# -----------------------------------------------------------
def _get_customs(compiler):
    custom_obj = {}
    if compiler.compiler['loss'] in CUSTOM_LOSES:
        custom_obj[compiler.compiler['loss']] = getattr(Sublosses, compiler.compiler['loss'])
    return custom_obj
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
