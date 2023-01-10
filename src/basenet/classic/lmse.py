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
        This is the BaseNetLMSE model. Implements a classic machine-learning LMSE.
        :param input_database: The BaseNetDatabase to train, validate and test the LMSE model.
        :param name: The name of the model.
        :param th: Validation threshold. Set up a threshold if you are working in a classification problem.
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
            true_path = f'{path}.npy'
        else:
            true_path = path
        np.save(true_path, self.weights)
        return true_path

    @staticmethod
    def load(path: str, name: str = 'unnamed_lmse'):
        """
        This function loads the weights of the LMSE model.
        :param path: Path where the weights are saved, or you can just import the weights.
        :param name: The name of the model.
        :return: The BaseNetLMSE with the given model path, without a linked database or results.
        """
        if isinstance(path, str):
            try:
                if '.npy' not in path:
                    true_path = f'{path}.npy'
                else:
                    true_path = path
                w = np.load(true_path)
            except Exception as ex:
                raise ValueError(f'BaseNetLMSE: The given path gave the following error while loading: {ex}.')
        elif isinstance(path, np.ndarray):
            w = path
        else:
            raise TypeError('BaseNetLMSE: Error while loading the model weights, it is not a path.')
        lmse = BaseNetLMSE(name=name)
        lmse.weights = w
        lmse.is_trained = True
        return lmse

    def transformation(self, original: bool = True, bias: bool = False):
        """
        This method returns the weights in their original shape.
        :param original: If true, returns the weights in the original shape, else, with the computational shape.
        :param bias: If bias is true, the bias is added to the weights.
        :return: The reshaped weights.
        """
        if bias:
            w = self.weights[1:]
        else:
            w = self.weights
        if original:
            return np.reshape(w, self.__shape)
        else:
            return w

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
        train = np.reshape(train, (matrix.shape[0], flatten_sape))
        return self.__add_bias(train)

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
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
