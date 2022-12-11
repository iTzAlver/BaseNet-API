# -------------------------------------------------------------------#
#                                                                    #
#    Author:    Alberto Palomo Alonso.                               #
#                                                                    #
#    Git user:  https://github.com/iTzAlver                          #
#    Email:     ialver.p@gmail.com                                   #
#                                                                    #
# -------------------------------------------------------------------#
from .database import BaseNetDatabase
from types import FunctionType
# -------------------------------------------------------------------#
#   x   x   x   x   x   x   x   x   x   x   x   x   x   x   x   x    #
# -------------------------------------------------------------------#


class BaseNetFeeder:
	def __init__(self, feeder, input_parameters: tuple = (), distribution=None,
	             name: str = 'unnamed_database', batch_size: int = None,
	             rescale: float = 1.0, dtype: tuple[str, str] = ('float', 'float'),
	             bits: tuple[int, int] = (32, 32)):
		"""
		The BaseNetFeeder class provides a wrapper to the BaseNetDatabase, but it takes a function as an input that
		generates the database. This is useful for mathemartical problems or problems that require some type of data
		ingestion.
		:param feeder: The feeder function that returns a tuple with "x" and "y" values for a synthetic dataset.
		:param input_parameters: The input parameters of the feeder function.
        :param distribution: The distribution of the datasets, default: {'train': 70, 'val': 20, 'test': 10}
        :param name: The database name.
        :param batch_size: Custom batch size for training.
        :param rescale: Rescale factor, all the values in x are divided by this factor, in case rescale is needed.
        :param dtype: Data type of the dataset. ('input', 'output') (x, y)
        :param bits: Bits used for the data type. ('input', 'output') (x, y)
		"""
		self.feeder = feeder
		self.__stored_parameters: tuple = input_parameters
		self.__db_options: tuple = (distribution, name, batch_size, rescale, dtype, bits)
		samp = feeder(*input_parameters)
		self.db: BaseNetDatabase = BaseNetDatabase(*samp,
		                                           distribution=distribution,
		                                           bits=bits,
		                                           name=name,
		                                           batch_size=batch_size,
		                                           rescale=rescale,
		                                           dtype=dtype)

	def __generate(self, *parameters):
		if not parameters:
			_parameters = self.__stored_parameters
		else:
			_parameters = parameters
		return BaseNetDatabase(*self.feeder(*_parameters),
		                       distribution=self.__db_options[0],
		                       name=self.__db_options[1],
		                       batch_size=self.__db_options[2],
		                       rescale=self.__db_options[3],
		                       dtype=self.__db_options[4],
		                       bits=self.__db_options[5])

	def __check(self, samp):
		if not isinstance(samp, tuple):
			raise ValueError(f'BaseNetFeeder:__init__: The return value of the {self.feeder.__name__} function is not a'
			                 f' tuple containing "x", "y" values...')
		if len(samp[0]) != len(samp[1]):
			raise ValueError(f'BaseNetFeeder:__init__: The return value of the {self.feeder.__name__}: the number of'
			                 f' samples of "x" is not equal to the number of samples of "y"...')
		if not isinstance(self.feeder, FunctionType):
			raise ValueError(f'BaseNetFeeder:__init__: The type of the feeder is not a fuction, it is a '
			                 f'{type(self.feeder)}, it must be a function returning "x" and "y" values...')

	def refresh(self, *parameters):
		"""
		Generates a new database with the given parameters.
		:param parameters: The parameters of the feeder function.
		:return: self
		"""
		self.db = self.__generate(*parameters)
		return self

	def __getattr__(self, item):
		__return_value = getattr(self.db, item)
		return __return_value

	def __repr__(self):
		return f'BaseNetFeeder:\n\tFeeder name: {self.feeder.__name__}.\n\tParameters: {self.__stored_parameters}\n\t' \
		       f'BaseNetDatabase: {self.db}.\n'
# -------------------------------------------------------------------#
#           E   N   D          O   F           F   I   L   E         #
# -------------------------------------------------------------------#
