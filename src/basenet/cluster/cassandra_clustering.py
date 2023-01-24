# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
import cassandra.cluster as cc
import uuid

CREATE_KEYSPACE = """CREATE KEYSPACE IF NOT EXISTS"""
CREATE_TABLE = """CREATE TABLE IF NOT EXISTS"""
INSERT_DATA = """INSERT INTO"""
EXTRACT_DATA = """SELECT"""
# -----------------------------------------------------------


class CassandraCluster:
    def __init__(self, ip_address: (list[str], str), port: str, database_name: str):
        """
        The CassandraCluster
        --------------------

        This class implements an API to interact with a Cassandra cluster.


        Methods
        -------
        The class instances can execute the following methods:

        * **set_keyspace**: This method changes the keyspace or creates it in the CQL query.
        * **set_table**: This method creates or access a table in the CQL query.
        * **transfer_data**: This method stores data in the database.
        * **get_data**: This method obtains the data in the database.
        * **execute**: This method executes a specific CQL query.

        :param ip_address: List of nodes to connect with the database.
        :param port: Connection port, usually 9042.
        :param database_name: The keyspace to access the data.
        """
        if not isinstance(ip_address, str) or not isinstance(port, str) or not isinstance(database_name, str):
            raise ValueError('Error in the input data, "ip_address", "database_name" and "port" must be strings.')
        self.keyspace = database_name

        self.table = None
        self.keyspace = None
        self.__format = None

        if isinstance(ip_address, str):
            self.__address = [ip_address]
        else:
            self.__address = [ip for ip in ip_address]
        self.__port = port

        self.__cluster = cc.Cluster(self.__address)
        self.__session = self.__cluster.connect()
        self.set_keyspace(database_name)

    def set_keyspace(self, keyspace: str):
        """
        This method sets a keyspace in the session. If it does not exist, it creates one.

        :param keyspace: The keyspace in the database.
        :return: The current object.
        """
        self.keyspace = keyspace
        command = f'{CREATE_KEYSPACE} {keyspace} WITH REPLICATION = ' \
                  r"{'class': " \
                  r"'SimpleStrategy'," \
                  r"'replication_factor': 3}"
        self.__session.execute(command)
        self.__session.set_keyspace(keyspace)
        return self

    def set_table(self, table_name: str, *column_format):
        """
        This method creates or access to the given table.

        Example
        -------
        If you want to create a table, this is an example:

            ``my_database.set_table(table_name='NewTable', ('id', 'uuid'), ('name', 'text'),
            ('date', 'text'), ('age', 'int'))``

        Where you create a table with the columns 'id', 'name', 'date', 'age'. **All tables MUST have the id parameter
        specified as uuid.**

        :param table_name: The table name in the database.
        :param column_format: The column format of the table **if it is going to be created** as tuples.
        :return: The current object.
        """
        self.table = table_name
        if column_format:
            self.__format = column_format
            _form = ''
            for index, zero_format in enumerate(column_format):
                if index == 0:
                    _form = f'\n{zero_format[0]} {zero_format[1]} PRIMARY KEY,\n'
                elif index == len(column_format) - 1:
                    _form = f'{_form}{zero_format[0]} {zero_format[1]}'
                else:
                    _form = f'{_form}{zero_format[0]} {zero_format[1]},\n'
            command = f'{CREATE_TABLE} {self.keyspace}.{table_name} ({_form})'
            self.__session.execute(command)
        return self

    def transfer_data(self, data: dict):
        """
        This method transfers data into the database in the current keyspace and table with the following format:

        ``{column_name_n: data_of_the_column_n, ...}``

        **Note that you don't have to specify the uuid as this method already does it inner.**

        :param data: A dictionary with a single data (keys are the columns and values are the current values).
        :return: The current object.
        """
        if len(data) > 0:
            _form = f'(id, '
            _s = '(%s, '
        else:
            raise ValueError('Cassandra error while transferring data: the data is empty.')
        for index, tag in enumerate(data):
            if index == len(data) - 1:
                _form = f'{_form}{tag})'
                _s = f'{_s}%s)'
            else:
                _form = f'{_form}{tag}, '
                _s = f'{_s}%s, '

        start_data = [uuid.uuid1()]
        start_data.extend([data[tag] for tag in data])
        _data = tuple(start_data)
        command = f'{INSERT_DATA} {self.keyspace}.{self.table} {_form} VALUES {_s}'
        self.__session.execute(command, _data)
        logging.info('CassandraCluster: The information was introduced successfully.')
        return self

    def get_data(self, conditions: str, field: str = '*', limit: int = None):
        """
        This method extracts data from the database.

        :param conditions: The conditions to be met in order to get the data.
        :param field: The columns to access.
        :param limit: A random selection limit of hits.
        :return: The result of the execution.
        """
        if not isinstance(limit, int):
            command = f'{EXTRACT_DATA} {field} FROM {self.keyspace}.{self.table} WHERE {conditions} ALLOW FILTERING;'
        else:
            command = f'{EXTRACT_DATA} {field} FROM {self.keyspace}.{self.table} WHERE {conditions} AND ' \
                      f'token(id) > token(now()) LIMIT {limit} ALLOW FILTERING;'
        return self.__session.execute(command)

    def execute(self, command: str):
        """
        This function executes a CQL command in the database.

        :param command: The current command.
        :return: The result of the execution.
        """
        return self.__session.execute(command)

    # ---------------------------------------------------------------------------------------------------------------
    # Context manager.
    # ---------------------------------------------------------------------------------------------------------------
    def close(self):
        """
        This method closes the connection with Cassandra.
        :return: Nothing.
        """
        try:
            if self.__session is not None and self.__cluster is not None:
                self.__session.shutdown()
            if self.__cluster is not None:
                self.__cluster.shutdown()
        except Exception as ex:
            raise RuntimeError(f'There was an error while closing the connection with Cassandra:\n{ex}.')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self):
        command_output = self.execute(f'DESCRIBE TABLES')
        coms = [command.name for command in command_output]
        _text = f'CassandraCluster instance to {self.__address}:{self.__port}:\n\tCurrent keyspace:' \
                f'\t{self.keyspace}\n\tCurrent table:\t\t{self.table}\n\nAvailable tables:\n{coms}\n'
        return _text


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                          MAIN                             #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# if __name__ == '__main__':
#     with CassandraCluster('192.168.79.48', '9042', 'geoimage').set_table('master') as database:
#         print(database)
#         results = database.get_data("country = 'Spain'", limit=5)
#         [print(_.continent) for _ in results]
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
