# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import numpy as np
from ..database import BaseNetDatabase


class BaseNetLMSE:
    def __init__(self, input_database: (BaseNetDatabase, str, None) = None, name: str = 'unnamed_lmse',
                 th: (None, float) = None):
        """
        BaseNetLMSE
        -----------

        This is the BaseNetLMSE model. Implements a classic machine-learning LMSE.

        :param input_database: The BaseNetDatabase to train, validate and test the LMSE model.
        :param name: The name of the model.
        :param th: Validation threshold. Set up a threshold if you are working in a classification problem.

        Attributes
        ----------
        The BaseNetLMSE has the following attributes:

        * linked_database: A BaseNetDatabase with train, validation and test datasets.
        * name: A string with the model's name.
        * th: A float telling the model threshold, if the output is binary, the threshold will be applied to non-binary
          outputs of the LMSE.
        * weights: A numpy.ndarray (2D) with the current trained weights of the model in its computational shape (2D).
        * is_trained: A boolean telling if the model is trained.
        * results: The MeanSquaredError, MeanAverageError and TotalErrorPerSample in a dictionary with the 'mse', 'mae'
          and 'error' keywords.

        Methods
        -------
        The BaseNetLMSE has the following methods:

        * link_database: Use this method to link a BaseNetDatabase to the model.
        * fit: Use this method to train the model. If a BaseNetDatabase is provided to the constructor, the fitting
          process and validation are automatic.
        * validate: Computes the 'mse', 'mae' and 'error' of the validation dataset. If a BaseNetDatabase is provided to
          the constructor, the fitting process and validation are automatic.
        * predict: Takes a sample as input and predicts the output with the trained values.
        * evaluate: Uses the test samples of the linked BaseNetDatabase to compute a provided metric as a function.
        * transformation: Returns the weights of the model as a linear transformation with or without the bias.
        * save_weights: Saves the weights and bias in the given path.
        * load: Builds a BaseNetLMSE from the given weights and bias (as a tuple or path).
        """
        # Database import.
        self.linked_database = None
        self.name = name
        self.th = th
        self.__shape = tuple()
        self.weights = None
        self.is_trained = False
        self.results = {'mse': None,
                        'mae': None,
                        'error': None}

        if input_database is not None:
            self.link_database(input_database)
            self.fit()
            self.is_trained = True
            self.validate(th)

    def link_database(self, input_database: (BaseNetDatabase, str)):
        """
        This method links a BaseNetDatabase to the model.
        :param input_database: The input BaseNetDatabase or a path where it is saved.
        :return: The object of the method.
        """
        if isinstance(input_database, str):
            database = BaseNetDatabase.load(input_database)
        elif isinstance(input_database, BaseNetDatabase):
            database = input_database
        else:
            raise TypeError('BaseNetLMSE:The input database to the LSTM is not a path or BaseNetDatabase.')
        self.linked_database = database
        return self

    def fit(self):
        """
        This method computes the weights of the BaseNetLMSE.
        :return: The computed weights of the LMSE matrix.
        """
        # Data extraction.
        if self.linked_database is not None:
            database = self.linked_database
        else:
            raise RuntimeError('BaseNetLMSE: Tried to train the model without a linked database. '
                               'Link a database with the "link_database()" method.')
        xtrain = self.__utility_conversion(database.xtrain)
        ytrain = database.ytrain
        # Train the matrix.
        v = self.__train_matrix(xtrain, ytrain)
        # Weights and shape.
        _shape = [shape for shape in database.xtrain.shape[1:]]
        if _shape[-1] == 1:
            _shape.pop()
        _shape.append(ytrain.shape[-1])
        self.__shape = tuple(_shape)
        self.weights = v
        return v

    def validate(self, th: (None, float) = None):
        """
        This method computes the errors in the validation dataset with the linked database.
        :param th: The threshold in the output, if provided.
        :return: The MeanSquaredError.
        """
        if self.linked_database is not None:
            database = self.linked_database
        else:
            raise RuntimeError('BaseNetLMSE: Tried a validation without a linked database. Link a database with the'
                               '"link_database()" method.')
        if self.is_trained is False:
            RuntimeError('BaseNetLMSE: Tried a validation without a trained model. Link a database with the'
                         '"link_database()" method and train it with the "fit()" method.')
        # Data extraction.
        xtest = database.xval
        ytest = database.yval
        # Validation.
        ytest_hat = self.predict(xtest, th)
        diff = ytest_hat - ytest
        mse = np.mean(diff**2)
        mae = np.mean(abs(diff))
        error = np.sum(abs(diff)) / len(ytest)
        self.results = {'mse': mse,
                        'mae': mae,
                        'error': error}

    def predict(self, x: (list, tuple, np.ndarray), th: (None, float) = None):
        """
        This method predicts the outputs using the current model weights.
        :param x: A list, tuple, structured array of inputs.
        :param th: The threshold in the output, if provided.
        :return: A list, tuple, structured array with the predicted values.
        """
        main_type = type(x)
        x_ = self.__utility_conversion(np.array(x))
        ytest_hat_float = np.matmul(x_, self.weights)
        if th is not None:
            y = self.__threshold(ytest_hat_float, th)
        elif self.th is not None:
            y = self.__threshold(ytest_hat_float, self.th)
        else:
            y = ytest_hat_float

        # Return the main type of x...
        if main_type is list:
            return list(y)
        elif main_type is tuple:
            return tuple(y)
        elif main_type is np.ndarray:
            return np.array(y)
        else:
            return None

    def evaluate(self, metric, th: (None, float) = None):
        """
        This method computes the errors in the test dataset with the linked database and the given function metric.
        :param metric: The metric function, taking as input: (the predicted value, the reference) -> metric_value.
        :param th: The threshold in the output, if provided.
        :return: The error measurement according to the metric.
        """
        if self.linked_database is not None:
            database = self.linked_database
        else:
            raise RuntimeError('BaseNetLMSE: Tried a test without a linked database. Link a database with the'
                               '"link_database()" method.')
        if self.is_trained is False:
            RuntimeError('BaseNetLMSE: Tried a test without a trained model. Link a database with the'
                         '"link_database()" method and train it with the "fit()" method.')
        # Data extraction.
        xtest = database.xtest
        ytest = database.ytest
        # Validation.
        ytest_hat = self.predict(xtest, th)
        metric_value = metric(ytest_hat, ytest)
        return metric_value

    def save_weights(self, path: str):
        """
        This method saves the weights of the model into the given path as a numpy array.
        :param path: String with the save path.
        :return: The path where it is finally saved.
        """
        if not isinstance(path, str):
            raise TypeError('BaseNetLMSE: Error while saving the model weights, the path must be a string.')
        if '.npy' not in path:
            path_weights = f'{path}_weights.npy'
            path_bias = f'{path}_bias.npy'
        else:
            path_weights = f"{path.replace('.npy', '')}_weights.npy"
            path_bias = f"{path.replace('.npy', '')}_bias.npy"
        np.save(path_weights, self.weights)
        np.save(path_bias, self.weights[0, :])
        return path_weights, path_bias

    @staticmethod
    def load(path: (str, tuple[np.ndarray]), name: str = 'unnamed_lmse', th: (None, float) = None):
        """
        This function loads the weights of the LMSE model.
        :param path: Path where the weights are saved, or you can just import the weights.
        :param name: The name of the model.
        :param th: The model threshold if it is a classification problem (binary output).
        :return: The BaseNetLMSE with the given model path, without a linked database or results.
        """
        if isinstance(path, str):
            try:
                if '.npy' not in path:
                    path_weights = f'{path}_weights.npy'
                    path_bias = f'{path}_bias.npy'
                else:
                    path_weights = f"{path.replace('.npy', '')}_weights.npy"
                    path_bias = f"{path.replace('.npy', '')}_bias.npy"
                w = np.load(path_weights)
                b = np.load(path_bias)
            except Exception as ex:
                raise ValueError(f'BaseNetLMSE: The given path gave the following error while loading: {ex}.')
        elif isinstance(path, tuple):
            w = path[0]
            b = path[1]
        else:
            raise TypeError('BaseNetLMSE: Error while loading the model weights and bias, it is not a path or tuple.')
        lmse = BaseNetLMSE(name=name)
        lmse.weights = np.concatenate([np.array([b]), w])
        lmse.is_trained = True
        lmse.th = th
        return lmse

    def transformation(self, original: bool = True, bias: bool = False):
        """
        This method returns the weights in their original shape.
        :param original: If true, returns the weights in the original shape, else, with the computational shape.
        :param bias: If bias is true, the bias is added to the weights.
        :return: The reshaped weights and the bias if requested.
        """
        w = self.weights[1:]
        b = self.weights[0]
        if original and not bias:
            return np.reshape(w, self.__shape)
        elif original and bias:
            return np.reshape(w, self.__shape), b
        elif not original and not bias:
            return w
        else:
            return w, b

    # Model functions.
    @staticmethod
    def __add_bias(matrix: np.ndarray):
        ones_tensor = np.ones(matrix.shape[:-1])
        biasing = np.expand_dims(ones_tensor, 1)
        return np.concatenate([biasing, matrix], axis=1)

    def __utility_conversion(self, matrix: np.ndarray):
        if matrix.shape[-1] == 1:
            train = np.squeeze(matrix, -1)
        else:
            train = matrix
        flatten_sape = 1
        for dimension in matrix.shape[1:]:
            flatten_sape *= dimension
        flat_train = np.reshape(train, (matrix.shape[0], flatten_sape))
        return self.__add_bias(flat_train)

    @staticmethod
    def __train_matrix(x: np.ndarray, y: np.ndarray):
        q_i = np.linalg.pinv(x.T)
        a = y.T @ q_i
        return a.T

    @staticmethod
    def __threshold(matrix: np.ndarray, th: float = 0.5):
        hat = matrix
        hat[hat >= th] = 1
        hat[hat < th] = 0
        return hat

    def __repr__(self):
        if self.__shape:
            __shape = self.__shape
        else:
            __shape = 'unknown shape'
        _text = f'\n=================================\n' \
                f'<BaseNetLMSE model: {self.name}>:\n' \
                f'\tTrained:\t{self.is_trained}\n' \
                f'\tShape:\t\t{__shape}\n' \
                f'\tBinary:\t\t{self.th is not None}\n' \
                f'=================================\n'
        return _text

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def __matmul__(self, other):
        return self.predict([other])
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
