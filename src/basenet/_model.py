# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import pickle
import os
import copy
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils.vis_utils import plot_model
from ._database import BaseNetDatabase
from .utils import StdoutLogger
from .algorithms import Subkeras
from .loss_functions import Sublosses
from ._names import KERAS_LIST_LAYERS, PREBUILT_LOSSES, PREBUILT_LAYERS
from .__special__ import __keras_checkpoint__, __tensorboard_logs__, __print_model_path__


# -----------------------------------------------------------
class BaseNetModel:
    def __init__(self, compiler=None, model: keras.Model = None, name: str = '', verbose: bool = False):
        """
        The BaseNetModel implements an API that makes use of keras and tensorflow to build Deep Learning Models.
        :param compiler: BaseNetCompiler object to build the model.
        :param model: If a keras.model is already compiled, you can import it in the model parameter, so the compiler
        won't be used during the construction.
        """
        self._verbose = verbose

        self.compiler = compiler
        self.is_trained = False
        self.name = name
        self.breech = []
        self.metrics = {'train': [], 'val': []}

        try:
            if model is not None:
                self.model = model
                if not self.name:
                    self.name = 'unnamed_model'
                self.is_compiled = True
                self.summary = model.summary()

            elif self.compiler is not None:
                self.model = self._build()
                if not self.name:
                    self.name = compiler.name
                self.is_compiled = True
                self.summary = self._get_summary()

            else:
                self.model = None
                self.is_compiled = False
                if not self.name:
                    self.name = 'uncompleted_model'
                self.summary = 'uncompleted_model'

        except Exception as ex:
            if not self.name:
                self.name = 'uncompleted_model'
            self.model = None
            self.is_compiled = False
            self.summary = ex
            print(f'BaseNetModel: The model is empty. Raised the following exception: {ex}.')

    def fit(self, ndb, epochs):
        """
        This function fits the BaseNetModel with the selected database.
        :param ndb: Index of the database already loaded.
        :param epochs: Number of epochs to train.
        :return: History of the fitting process.
        """
        if ndb < len(self.breech):
            db = self.breech[ndb]
        else:
            if self._verbose:
                print('BaseNetModel: Cannot load the BaseNetDatabase to fit, the index of the database does not exist.')
            return None
        xtrain = db.xtrain
        ytrain = db.ytrain
        xval = db.xval
        yval = db.yval
        history = None
        try:
            history = self.model.fit(xtrain, ytrain, batch_size=db.batch_size, epochs=epochs,
                                     validation_data=(xval, yval),
                                     callbacks=[
                                         self._FitCallback(self),  # TODO
                                         tf.keras.callbacks.TensorBoard(log_dir=__tensorboard_logs__),
                                         tf.keras.callbacks.EarlyStopping(patience=6),
                                         tf.keras.callbacks.ModelCheckpoint(filepath=__keras_checkpoint__)
                                     ])
            self.metrics['val'].extend(history.history['val_loss'])
            self.metrics['train'].extend(history.history['loss'])

        except Exception as ex:
            if self._verbose:
                print(f'BaseNetModel: Cannot train the model, it raised an exception: {ex}.')
        finally:
            return history

    def predict(self, x, scale: float = 1.0, th: (None, float) = 0.5, expand_dims: bool = False) -> tf.Tensor:
        """
        This function predicts with the current model an output from the input 'x', divided by the 'scale' and converted
        to a binary matrix with a custom threshold 'th'.
        :param x: Input np.array o tf.Tensor.
        :param scale: Divides by a custom scale (default: 255.0 -> 8bit images).
        :param th: Custom output threshold (default: 0.5 -> mid-range predictions). Set up to 'None' to see
        the real output.
        :param expand_dims: Expands the dimension of the tensor.
        :return: The prediction output of the model.
        """
        _x_ = tf.convert_to_tensor(np.array(x).astype("float32") / scale)
        if expand_dims:
            __x = tf.expand_dims(_x_, axis=-1)
        else:
            __x = _x_
        _y_ = self.model.predict(__x)
        if th is not None:
            __y__ = self._threshold(_y_, th)
        else:
            __y__ = _y_
        return __y__

    def evaluate(self, ndb, metric, th: (None, float) = 0.5) -> (tf.Tensor, None):
        """
        This method evaluates the test dataset of a BaseNetDatabase.
        :param ndb: Number of the database to be tested.
        :param metric: A metric function.
        :param th: The threshold of the prediction, if not None.
        :return: The evaluated metric.
        """
        if ndb < len(self.breech):
            db = self.breech[ndb]
        else:
            if self._verbose:
                print('BaseNetModel: Cannot load the BaseNetDatabase to evaluate, '
                      'the index of the database does not exist.')
            return None
        xtest = db.xtest
        ytest = db.ytest
        _output_ = self.predict(xtest, th=th, scale=1.0)
        result = metric(_output_, ytest)
        return result

    def add_database(self, db: (BaseNetDatabase, None) = None, db_path: str = '') -> object:
        """
        This method adds a database into the model.
        :param db: A BaseNetDatabase.
        :param db_path: A path to a BaseNetDatabase.
        :return: The same object.
        """
        try:
            if db is not None:
                self.breech.append(db)
            elif db_path:
                self.breech = BaseNetDatabase.load(db_path)
            else:
                if self._verbose:
                    print('BaseNetModel: Cannot load the BaseNetDatabase: there is no path or model provided.')
        except Exception as ex:
            print(f'BaseNetModel: Cannot load the BaseNetDatabase, an exception raised: {ex}.')
        finally:
            return self

    # Save and load methods:
    def save(self, model_path: str, compiler_path: str = '') -> bool:
        """
        This function saves the BaseNetModel in a pair: .cpl (BaseNetCompiler) and .h5 (keras.model) format.
        :param model_path: Path where the keras.model is being saved in the file system.
        :param compiler_path: Path where the BaseNetCompiler is being saved in the file system.
        :return: True if the saving was successful. False if not.
        """
        try:
            if self.model is not None:
                self.model.save(model_path)
            else:
                if self._verbose:
                    print('BaseNetModel: Cannot save the model because it is not compiled yet.')
                return False

            if self.compiler:
                if not compiler_path:
                    _compiler_path = model_path.replace('.h5', '.cpl')
                elif '.cpl' in compiler_path:
                    _compiler_path = compiler_path
                else:
                    _compiler_path = f'{compiler_path}.cpl'
                self.compiler.save(_compiler_path)
            return True

        except Exception as ex:
            if self._verbose:
                print(f'BaseNetModel: Cannot save the model because a exception was raised: {ex}.')
            return False

    @staticmethod
    def load(model_path: str, compiler_path: str = '') -> object:
        """
        This function loads a pair: .cpl (BaseNetCompiler) and .h5 (keras.model) format and builds a BaseNetModel from
        the loaded parameters.
        :param model_path: Path where the keras.model is being loaded from the file system.
        :param compiler_path: Path where the BaseNetCompiler is being loaded from the file system.
        :return: The BaseNetModel with the given model path.
        """
        if not compiler_path:
            _compiler_path = model_path.replace('.h5', '.cpl')
        elif '.cpl' in compiler_path:
            _compiler_path = compiler_path
        else:
            _compiler_path = f'{compiler_path}.cpl'
        if os.path.exists(_compiler_path):
            with open(_compiler_path, 'rb') as file:
                compiler = pickle.load(file)
        else:
            print('BaseNetModel: The compiler path is empty, the current model has no compiler.')
            compiler = None

        try:
            model = keras.models.load_model(model_path)
        except Exception as ex:
            print(f'BaseNetModel: The model raised an exception: {ex}.')
            model = None

        name = model_path.split('/')[-1].replace('.h5', '')
        return BaseNetModel(compiler, model=model, name=name)

    def print(self, print_path: str = __print_model_path__) -> object:
        """
        This function renders an image with the architecture of the compiled model.
        :param print_path: Path where the image of the model is saved.
        :return: A bypass of the current object.
        """
        try:
            if self.model is not None:
                plot_model(self.model, to_file=f'{print_path}/{self.name}.png', show_shapes=True)
            else:
                if self._verbose:
                    print('BaseNetModel: Cannot print the model, the model is empty.')
        except Exception as ex:
            if self._verbose:
                print(f'BaseNetModel: The compiler path is empty, the model raised the following exception: {ex}.')
        finally:
            return self

    # Private methods:
    def _build(self) -> keras.Model:
        _scope = self._get_scope(self.compiler.devices)

        with _scope:
            # Add the input of the model.
            _inp = keras.Input(shape=self.compiler.io_shape[0])
            _inp._name = 'compiled-model-keras'
            _lastlay = _inp

            last_master = None
            pipeline_opened = False
            towers = []

            for layer in self.compiler.layers:
                for layer_type, (layer_shape, layer_args) in layer.items():
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
                        if layer_type in KERAS_LIST_LAYERS and layer_type not in PREBUILT_LAYERS:
                            this_lay = getattr(keras.layers, layer_type)(*layer_shape, **layer_args)
                        elif layer_type in KERAS_LIST_LAYERS:
                            this_lay = getattr(Subkeras, layer_type)(*layer_shape, **layer_args)
                        else:
                            importmodel = keras.models.load_model(f'{layer_type}.h5')
                            importmodel._name = layer_type
                            this_lay = importmodel

                        _lastlay = this_lay(_lastlay)

            # Add the output of the model.
            out = keras.layers.Dense(self.compiler.io_shape[1], activation="sigmoid", name='output')(_lastlay)
            _compile = copy.copy(self.compiler.compile_options)

            if _compile['loss'] in PREBUILT_LOSSES:
                _compile['loss'] = getattr(Sublosses, self.compiler.compile_options['loss'])

            model = keras.Model(_inp, out, name=self.name)
            model.compile(**_compile)
        return model

    def _get_summary(self):
        log = StdoutLogger()
        log.start()
        self.model.summary()
        __msg = log.messages
        _msg = ''
        for msg in __msg:
            _msg = f'{_msg}{msg}'
        summary = _msg
        log.stop()
        log.flush()
        return summary

    @staticmethod
    def _get_scope(devices):
        # Creates a scope from the current devices.
        _devices = []
        for _dev, role in devices.items():
            if role == 'Train':
                _devices.append(_dev)
        if len(_devices) > 0:
            strategy = tf.distribute.MirroredStrategy(devices=_devices)
            scope = strategy.scope()
        else:
            raise ValueError('There are no training devices...')
        return scope

    class _FitCallback(keras.callbacks.Callback):
        # TODO
        def __init__(self, master):
            self.master = master
            super().__init__()

        def on_epoch_begin(self, epoch, logs=None):
            keys = list(logs.keys())
            print(f"Start epoch {epoch} of training; got log keys: {keys}")

        def on_epoch_end(self, epoch, logs=None):
            keys = list(logs.keys())
            print(f"End epoch {epoch} of training; got log keys: {keys}")
        pass

    @staticmethod
    def _threshold(y, th):
        __y__ = []
        for element in y:
            if element > th:
                __y__.append(1)
            else:
                __y__.append(0)
        return tf.convert_to_tensor(__y__)

    # Build functions:
    def __repr__(self):
        return f'Model object with the following parameters:\nCompiler: {self.compiler}\nSummary: {self.summary}'

    def __bool__(self):
        return self.is_compiled

    def __sizeof__(self):
        return len(self.compiler.layers)

    def __eq__(self, other):
        return self.compiler.layers == other.compiler.layers
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
