# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
"""
The model.py file includes the BaseNetModel class and the BaseNetResults class.
"""
# Import statements:
import pickle
import os
import copy
import shutil
import webbrowser
import logging
import numpy as np
import tensorflow as tf

from multiprocessing import Process, Queue
from tensorboard import program
from tensorflow import keras
from keras.utils.vis_utils import plot_model

from ._utils import StdoutLogger
from ._algorithms import Subkeras
from ._loss_functions import Sublosses

from ._names import KERAS_LIST_LAYERS, PREBUILT_LOSSES, PREBUILT_LAYERS
from .__special__ import __keras_checkpoint__, __tensorboard_logs__, __print_model_path__, __bypass_path__

from .database import BaseNetDatabase


# -----------------------------------------------------------
class BaseNetModel:
    """
    The BaseNetModel class provides a wrapper for the tf.keras.model API with easier use. When initialized,
    it initializes a breech of databases in its attribute 'breech'. If we provide a compiler, the model will be
    built from the compiler; however, if we provide a tf.keras.model, the compiler is ignored and the model is built
    from the provided tf.keras.model.

    To add a database to the model, we can use the method BaseNetModel.add_database() that takes a BaseNetDatabase as
    input.

    The class contains load and save methods to store the compilers (.cpl files) and models (.h5 files) in the same
    directory.

    We also provide a BaseNetModel.fit() method that can create a separate process for training. The original framework
    does not include this feature:
        *   The BaseNetModel.fit() method takes as input the index of the loaded database via
        BaseNetModel.add_database() method and takes the train and validation subsets to fit the model.
        *   If the training process should not block the main process, the parameters 'avoid_lock' must be set to True,
        in that case, another process will take over the fitting tf.keras.model.fit() method and the information will
        be updated in the return class: BaseNetResults.
        *   In case we avoid the main process to be locked with the 'avoid_lock' feature, we will need to recover the
        tf.keras.model with the BaseNetModel.recover() method once the training is finished (check
        BaseNetResults.is_training).

    We can also evaluate the performance of the database with the BaseNetModel.evaluate() method, that makes use of the
    test subset.

    We can also predict the output of a certain input with the BaseNetModel.predict() method.

    We can also visualize the model with the BaseNetModel.print() method in a PNG image.
    """
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
            logging.error(f'BaseNetModel: The model is empty. Raised the following exception: {ex}.')

    def fit(self, ndb, epochs, tensorboard: bool = True, avoid_lock: bool = False):
        """
        This function fits the BaseNetModel with the selected database.
        :param ndb: Index of the database already loaded.
        :param epochs: Number of epochs to train.
        :param tensorboard: Activates or deactivates the Tensorboard.
        :param avoid_lock: Avoids the training process to lock the parent process.
        :return: BaseNetResults of the fitting process.
        """
        if tensorboard:
            self._flush()
            tb = program.TensorBoard()
            tb.configure(argv=[None, '--logdir', f'{__tensorboard_logs__}/{self.name}'])
            url = tb.launch()
            webbrowser.open(url, new=2)

        if ndb < len(self.breech):
            db = self.breech[ndb]
        else:
            if self._verbose:
                logging.warning('BaseNetModel: Cannot load the BaseNetDatabase to fit, the index of the '
                                'database does not exist.')
            return None

        xtrain = db.xtrain
        ytrain = db.ytrain
        xval = db.xval
        yval = db.yval

        __history__ = None
        fit_callback = _FitCallback()
        try:
            if avoid_lock:
                keras.models.save_model(self.model, __bypass_path__)
                queue = Queue()
                p = Process(target=self._fit_in_other_process, args=((xtrain, ytrain), (xval, yval),
                                                                     epochs, db.batch_size, self.name, queue, db.dtype))
                p.start()
                __history__ = BaseNetResults(queue=queue, parent=p)
            else:

                # Auto shard options. Avoid console-vomiting in TF 2.0.
                options = tf.data.Options()
                options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
                # Re-formatting the database.
                _xtrain = tf.convert_to_tensor(xtrain, dtype=getattr(tf, db.dtype[0]))
                _ytrain = tf.convert_to_tensor(ytrain, dtype=getattr(tf, db.dtype[1]))
                _xval = tf.convert_to_tensor(xval, dtype=getattr(tf, db.dtype[0]))
                _yval = tf.convert_to_tensor(yval, dtype=getattr(tf, db.dtype[1]))
                trai = tf.data.Dataset.from_tensor_slices((_xtrain, _ytrain)).batch(db.batch_size).with_options(options)
                val = tf.data.Dataset.from_tensor_slices((_xval, _yval)).batch(db.batch_size).with_options(options)

                history = self.model.fit(trai, batch_size=db.batch_size, epochs=epochs,
                                         validation_data=val,
                                         callbacks=[
                                             fit_callback,
                                             tf.keras.callbacks.TensorBoard(log_dir=f'{__tensorboard_logs__}/'
                                                                                    f'{self.name}'),
                                             tf.keras.callbacks.EarlyStopping(patience=10),
                                             tf.keras.callbacks.ModelCheckpoint(filepath=f'{__keras_checkpoint__}'
                                                                                         f'{self.name}.h5')
                                         ])
                __history__ = BaseNetResults(history.history['loss'], history.history['val_loss'])

        except Exception as ex:
            if self._verbose:
                logging.error(f'BaseNetModel: Cannot train the model, it raised an exception: {ex}.')
        finally:
            return __history__

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
                logging.warning('BaseNetModel: Cannot load the BaseNetDatabase to evaluate, '
                                'the index of the database does not exist.')
            return None
        xtest = tf.convert_to_tensor(db.xtest, dtype=getattr(tf, db.dtype[0]))
        ytest = tf.convert_to_tensor(db.ytest, dtype=getattr(tf, db.dtype[1]))
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
                self.breech.append(BaseNetDatabase.load(db_path))
            else:
                if self._verbose:
                    logging.warning('BaseNetModel: Cannot load the BaseNetDatabase: there is no path or model '
                                    'provided.')
        except Exception as ex:
            logging.error(f'BaseNetModel: Cannot load the BaseNetDatabase, an exception raised: {ex}.')
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
                    logging.warning('BaseNetModel: Cannot save the model because it is not compiled yet.')
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
                logging.error(f'BaseNetModel: Cannot save the model because a exception was raised: {ex}.')
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
            logging.warning('BaseNetModel: The compiler path is empty, the current model has no compiler.')
            compiler = None

        try:
            model = keras.models.load_model(model_path)
        except Exception as ex:
            logging.error(f'BaseNetModel: The model raised an exception: {ex}.')
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
                    logging.warning('BaseNetModel: Cannot print the model, the model is empty.')
        except Exception as ex:
            if self._verbose:
                logging.error(f'BaseNetModel: The compiler path is empty, the model raised the following '
                              f'exception: {ex}.')
        finally:
            return self

    def recover(self):
        """
        This functions recovers the model from a training when the option 'avoid_lock == True'.
        :return: True if there was a recover. False if there was not a recover or an exception raised.
        """
        try:
            if self.model is None:
                if os.path.exists(__bypass_path__):
                    self.model = keras.models.load_model(__bypass_path__)
                    os.remove(__bypass_path__)
                    return True
                else:
                    logging.error(f'BaseNetModel: The bypass path is empty.')
                    return False
            else:
                return False
        except Exception as ex:
            logging.error(f'BaseNetModel: An exception when recovering the model raised: {ex}')
            return False

    def call(self, *args, **kwargs):
        """
        When calling the class BaseNetModel, you can merge two different models. So the model1(model2) will merge the
        model2 into the model1. By default, all the model parameters and options are inherited from the model1.

        :param args: Incoming model; BaseNetModel or tf.keras.model.
        :param kwargs: {'name': the model name; inherits the model1 name by default,
        'parallel': True for separate inputs, False to be sequential; False by default,
        'options': compile options; inherits the model1 compile options by default}
        :return: A bypass of the current object.
        """
        if len(args) != 1:
            if self._verbose:
                logging.warning('BaseNetModel: The number of arguments to the callable class must be 1.')
            return self

        input_model = args[0]
        if not isinstance(input_model, (keras.models.Model, BaseNetModel)):
            if self._verbose:
                logging.warning(f'BaseNetModel: The input model must be a BaseNetModel or a tf.keras.model not a '
                                f'{type(input_model)}.')
            return self
        else:
            if isinstance(input_model, BaseNetModel):
                keras_model = input_model.model
            else:
                keras_model = input_model

        the_model_is_parallel = False
        the_model_is_under_the_current_model = True
        name = self.name
        options = self.compiler.compile_options
        for key, item in kwargs.items():
            if key == 'parallel':
                the_model_is_parallel = item
            elif key == 'top':
                the_model_is_under_the_current_model = item
            elif key == 'name':
                name = item
            elif key == 'options':
                options = item

        if not the_model_is_parallel:
            if the_model_is_under_the_current_model:
                result_model_layers = []
                result_model_layers.extend(self.model.layers)
                result_model_layers.extend(keras_model.layers[1:])
            else:
                result_model_layers = []
                result_model_layers.extend(keras_model.layers)
                result_model_layers.extend(self.model.layers[1:])

            _in = result_model_layers[0]
            mids = _in
            for layer in result_model_layers:
                mids = layer(mids)
            _out = mids
            result_model = keras.models.Model(inputs=[_in], outputs=[_out], name=name)

        else:
            _in0 = keras.Input(shape=self.model.input_shape)
            _in1 = keras.Input(shape=keras_model.input_shape)
            _out = keras.layers.concatenate([_in0, _in1])
            result_model = keras.models.Model(inputs=[(_in0, _in1)], outputs=[_out], name=name)

        result_model.compile(options)
        self.model = result_model
        return self

    # Private methods:
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

    @staticmethod
    def _threshold(y, th):
        __y__ = []
        for element in y:
            if element > th:
                __y__.append(1)
            else:
                __y__.append(0)
        return tf.convert_to_tensor(__y__)

    def _flush(self):
        flush_checkpoints = f'{__keras_checkpoint__}{self.name}.h5'
        flush_logs = f'{__tensorboard_logs__}/{self.name}/'
        if os.path.exists(flush_logs):
            shutil.rmtree(flush_logs)
        if os.path.exists(flush_checkpoints):
            os.remove(flush_checkpoints)

    @staticmethod
    def _fit_in_other_process(train, val, epochs: int, batch_size: int, name: str, queue: Queue,
                              dtype: tuple[str, str]):
        print('Joined other process for training.')
        model = keras.models.load_model(__bypass_path__)
        # Auto shard options. Avoid console-vomiting in TF 2.0.
        _xtrain = tf.convert_to_tensor(train[0], dtype=getattr(tf, dtype[0]))
        _ytrain = tf.convert_to_tensor(train[1], dtype=getattr(tf, dtype[1]))
        _xval = tf.convert_to_tensor(val[0], dtype=getattr(tf, dtype[0]))
        _yval = tf.convert_to_tensor(val[1], dtype=getattr(tf, dtype[1]))
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        # Re-formatting the database.
        train = tf.data.Dataset.from_tensor_slices((_xtrain, _ytrain)).batch(batch_size).with_options(options)
        val = tf.data.Dataset.from_tensor_slices((_xval, _yval)).batch(batch_size).with_options(options)

        fit_callback = _FitCallback(queue=queue)
        model.fit(train, batch_size=batch_size, epochs=epochs,
                  validation_data=val,
                  callbacks=[
                     fit_callback,
                     tf.keras.callbacks.TensorBoard(log_dir=f'{__tensorboard_logs__}/{name}'),
                     tf.keras.callbacks.EarlyStopping(patience=10),
                     tf.keras.callbacks.ModelCheckpoint(filepath=f'{__keras_checkpoint__}{name}.h5')
                  ])
        keras.models.save_model(model, __bypass_path__)

    # Build functions:
    def __repr__(self):
        return f'Model object with the following parameters:\nCompiler: {self.compiler}\nSummary: {self.summary}'

    def __bool__(self):
        return self.is_compiled

    def __sizeof__(self):
        return len(self.compiler.layers)

    def __eq__(self, other):
        return self.compiler.layers == other.compiler.layers

    def __call__(self, *args, **kwargs):
        """
        When calling the class BaseNetModel, you can merge two different models. So the model1(model2) will merge the
        model2 into the model1. By default, all the model parameters and options are inherited from the model1.

        :param args: Incoming model: BaseNetModel or tf.keras.model.
        :param kwargs: {'name': the model name: inherits the model1 name by default,
        'parallel': True for separate inputs, False to be sequential: False by default,
        'options': compile options: inherits the model1 compile options by default}
        :return: A bypass of the current object, even if an exception occurs.
        """
        try:
            return self.call(*args, **kwargs)
        except Exception as ex:
            if self._verbose:
                logging.error(f'BaseNetModel: Exception raised while evaluating the function BaseNetModelObject(): '
                              f'{ex}')
            return self
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF SUPERCLASS                  #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #


class _FitCallback(keras.callbacks.Callback):
    def __init__(self, queue: Queue = None):
        super().__init__()
        self.loss = []
        self.val_loss = []
        self.is_training = True
        self.queue = queue

    def on_train_begin(self, logs=None):
        self.loss = []
        self.val_loss = []
        self.is_training = True

    def on_batch_end(self, batch, logs=None):
        loss = logs.get('loss')
        val_loss = logs.get('loss')
        self.loss.append(loss)
        self.val_loss.append(val_loss)
        if self.queue:
            self.queue.put((loss, val_loss))

    def on_train_end(self, logs=None):
        self.is_training = False
        if self.queue:
            self.queue.put('END')
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF SUPERCLASS                  #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #


class BaseNetResults:
    """
    The class BaseNetResults is a data collector of a training process. Using the get() method you update the
    .is_training attribute and collect the information of the training process. If you are training in a separate
    process, consider using this structure in your code:

        results = my_basenet_model.fit(*args, **kwargs)
        while results.is_training:
            do_my_main_task()
            results_in_a_dictionary = results.get()
        my_basenet_model.recover()

        keep_doing_my_main_task()
    """
    def __init__(self, loss=None, val_loss=None, queue: Queue = None, parent: Process = None):
        """
        The BaseNetResults constructor should not be used by the default user.
        :param loss: __inner parameter__
        :param val_loss: __inner parameter__
        :param queue: __inner parameter__
        :param parent: __inner parameter__
        """
        self.is_training = True
        if loss:
            self._loss = loss
        else:
            self._loss = []
        if val_loss:
            self._val_loss = val_loss
        else:
            self._val_loss = []
        self._queue = queue
        self._parent = parent

    def get(self):
        """
        The get() method obtains the results from the training process.
        :return: A dictionary with the training and validation losses. {'loss': [], 'val_loss': []}
        """
        if self._queue:
            while not self._queue.empty():
                recover = self._queue.get()
                if isinstance(recover, str):
                    if recover == 'END':
                        self.is_training = False
                        self._parent.join()
                else:
                    self._loss.append(recover[0])
                    self._val_loss.append(recover[1])
        else:
            self.is_training = False
        return {'loss': self._loss, 'val_loss': self._val_loss}
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
