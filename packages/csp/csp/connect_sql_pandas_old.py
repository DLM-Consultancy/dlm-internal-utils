import pandas as pd
import pyodbc
import datetime
import numpy as np
import urllib
import sqlalchemy as sa
import logging

sql_pd_dtype_dict = {'int': 'int64',
                     'varchar': 'object',
                     'float': 'float64',
                     'datetime': 'datetime64[ns]'}

pd_sql_dtype_dict = {v: k for k, v in sql_pd_dtype_dict.items()}

# you can add your dtype below
varchar_equivalents = ['ntext', 'nvarchar', 'char']
int_equivalents = ['smallint', 'bigint', 'int identity', 'bit']
float_equivalents = ['decimal', 'numeric']
datetime_equivalents = ['datetime64[ns, Asia/Jakarta]']

# this inverts the pd_sql_dtype_dict


def dtype_sql_to_pd(d_type_sql):
    if d_type_sql in varchar_equivalents:
        d_type_sql = 'varchar'

    if d_type_sql in int_equivalents:
        d_type_sql = 'int'

    if d_type_sql in float_equivalents:
        d_type_sql = 'float'

    if d_type_sql in datetime_equivalents:
        d_type_sql = 'float'
    return sql_pd_dtype_dict[d_type_sql]


def pd_dtype_to_python_dtype(val):
    val_type = type(val)
    if val_type == np.int64:
        return int(val)
    elif val_type == np.float64:
        return float(val)
    elif val_type == np.datetime64:
        return npdt64_to_dt(val)
    elif val_type == str:
        return str(val)
    elif val_type == pd._libs.tslibs.timestamps.Timestamp:
        return val.to_pydatetime()
    else:
        raise Exception('data type of {} not recognized'.format(val_type))


def npdt64_to_dt(dt64):
    """takes a numpy datetime64[ns] format, and returns a datetime.datetime format"""
    ts = (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    return datetime.datetime.fromtimestamp(ts)


def columns_gen_sql(columns_list: list, with_value_str=True) -> str:
    values_str = '?'
    columns_str = columns_list[0]
    for i in range(len(columns_list)-1):
        values_str += ",?"
        columns_str += ", "+columns_list[i+1]
    if with_value_str:
        return columns_str, values_str
    else:
        return columns_str


def insert_str_gen_sql(table: str, columns_str: str, values_str: str, schema: str = 'dbo') -> str:
    return "insert into [{}].[{}] ({}) values ({})".format(schema, table, columns_str, values_str)


def return_frame_string(df):
    """returns a frame string and values string from a dataframe
    this is used for joining table using a query

    Args:
        df ([pandas dataframe]): [description]
        
    returns frame string, and values string
    
    df = pd.DataFrame([["qwer","ty"],["as","df"]], columns = ['testing', 'testing2'])
    
    example frame string: 'frame (testing, testing2)'
    example values string: 'VALUES (qwer, ty), (as, df)'
    """

    columns_list = list(df.columns)
    columns_str = "frame ("
    for column in columns_list:
        columns_str += str(column) + ", "
    columns_str = columns_str[:-2]
    columns_str += ")"

    df_list = df.values.tolist()
    values_str = "VALUES" + " "
    for i_row in df_list:
        values_str += "("
        for val in i_row:
            values_str += str(val) + ", "
        values_str = values_str[:-2]
        values_str += "), "
    values_str = values_str[:-2]

    return columns_str, values_str


def return_update_string(table, pd_series, where_clause="", schema='dbo'):
    # semua tipe yang harus diberikan petik
    to_str_list = [str]  # in datatype
    # to datetime string
    to_dt_str_list = [pd._libs.tslibs.timestamps.Timestamp]
    to_null_list = [pd._libs.tslibs.nattype.NaTType]
    columns_list = list(pd_series.index)
    values_list = list(pd_series)
    update_str = f'UPDATE {schema}.{table}'
    update_str += "\nSET "
    for i in range(len(columns_list)):
        value = values_list[i]

        # Check for NULL values first (including np.nan, None, and NaT)
        if pd.isna(value) or value is None:
            value = 'NULL'
        # Handle datetime values
        elif type(value) in to_dt_str_list:
            value = str(value)
            value = round_seconds_datetime64_ns(value)
            value = "\'" + value + "\'"
        # Handle string values
        elif type(value) in to_str_list:
            value = "\'" + str(value) + "\'"
        # Handle other types (numbers etc)
        else:
            value = str(value)

        update_str += "{} = {}".format(columns_list[i], value)
        update_str += ", "
    update_str = update_str[:-2]

    update_str += "\n" + where_clause
    update_str += ";\n\n"
    return update_str


class Sql_pd_cnxn:
    """[this is a class to create a connection and do some operations between python and an sql server, using the pandas and pyodbc package]
    Main Methods:
    1. get_df
    2. return_sql_table_columns_info_df
    3. insert_df_to_table
    """

    def __init__(self,
                 server,
                 database,
                 username,
                 password,
                 driver='{ODBC Driver 18 for SQL Server}',
                 verbose=True):

        params = urllib.parse.quote_plus(
            'DRIVER={ODBC Driver 18 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD=' + password + ';Connection Timeout=30;' + 'TrustServerCertificate=yes;' + 'MultipleActiveResultSets=True;')

        try:
            self.cnxn = pyodbc.connect(
                'DRIVER=' + driver + ';SERVER=' + server + ';DATABASE=' + database+';UID=' + username + ';PWD=' + password + ';Connection Timeout=30;' + 'TrustServerCertificate=yes;' + 'MultipleActiveResultSets=True;')

            self.engine = sa.create_engine(
                "mssql+pyodbc:///?odbc_connect=%s" % params)
            if verbose:
                logging.info('Connection established to {}/{}!'.format(server, database))

        except Exception as e:
            logging.info(e)

    def validate_df_columns(self, df, schema, table):
        cursor = self.cnxn.cursor()
        for col in list(df.columns):
            col_dtype_pd = str(df[col].dtype)
            for row in cursor.columns(table=table, schema=schema, column=col):
                col_dtype_sql = row.type_name
                if pd_sql_dtype_dict[col_dtype_pd] == 'float':
                    pd_sql_dtype_dict[col_dtype_pd] == 'float' or pd_sql_dtype_dict[col_dtype_pd] == col_dtype_sql == 'decimal'
                    
                elif pd_sql_dtype_dict[col_dtype_pd] == 'int':
                    pd_sql_dtype_dict[col_dtype_pd] == 'int' or pd_sql_dtype_dict[col_dtype_pd] == col_dtype_sql == 'bigint'

                elif pd_sql_dtype_dict[col_dtype_pd] == 'varchar':
                    pd_sql_dtype_dict[col_dtype_pd] == 'varchar' or pd_sql_dtype_dict[col_dtype_pd] == col_dtype_sql == 'char'
                    
                else:    
                    assert pd_sql_dtype_dict[col_dtype_pd] == col_dtype_sql, 'mis match dtype for col: {} pd: {} != sql: {}'.format(col,
                                                                                                                                pd_sql_dtype_dict[col_dtype_pd], col_dtype_sql)
        logging.info('Data types and columns are matched!\n')
        cursor.close()

    def return_sql_table_columns_info_df(self, table: str, schema: str = 'dbo') -> pd.core.frame.DataFrame:
        """[returns the sql table columns information as a pandas dataframe..]

        Args:
            table (str): [description]
            schema (str, optional): [description]. Defaults to 'dbo'.

        Returns:
            pd.core.frame.DataFrame: [description]
        """
        cursor = self.cnxn.cursor()
        col_info = cursor.columns(table=table)
        table_info_arr = [[
            column.table_cat,
            column.table_name,
            column.column_name,
            column.type_name,
            column.column_size,
            column.buffer_length,
            column.decimal_digits,
            column.num_prec_radix,
            column.nullable,
            column.remarks,
            column.column_def,
            column.sql_data_type,
            column.sql_datetime_sub,
            column.char_octet_length,
            column.ordinal_position,
            column.is_nullable
        ] for column in col_info]
        cursor.close()
        df_columns = ['table_cat',
                      'table_name',
                      'column_name',
                      'type_name',
                      'column_size',
                      'buffer_length',
                      'decimal_digits',
                      'num_prec_radix',
                      'nullable',
                      'remarks',
                      'column_def',
                      'sql_data_type',
                      'sql_datetime_sub',
                      'char_octet_length',
                      'ordinal_position',
                      'is_nullable']
        table_info_df = pd.DataFrame(table_info_arr, columns=df_columns)
        return table_info_df

    def get_df(self, table: str = None, schema: str = 'dbo',  select_all=True, n_top_rows: int = None, columns_list: list = None, order_by: str = None, descending=False, where_clause: str = None, query=None, verbose=False) -> pd.core.frame.DataFrame:
        """
        This function returns a pandas dataframe from an SQL Table
        (!) if the arg 'select_all' is True, the other filter args will not work
        (!) the data type of columns from the returned dataframe with similar datatypes

        Args:
            table (str): [description]
            schema (str, optional): [description]. Defaults to 'dbo'.
            select_all (bool, optional): [description]. Defaults to True.
            n_top_rows (int, optional): [description]. Defaults to None.
            columns_list (list, optional): [description]. Defaults to None.
            order_by (str, optional): [description]. Defaults to None.
            descending (bool, optional): [description]. Defaults to False.
            query (str, optional): [description]. Defaults to None. basically, if you input the query directly, it will overide the other query fields, and use this query instead..

        Returns:
            pd.core.frame.DataFrame: [description]
        """
        if not query:
            if select_all:

                if columns_list:
                    columns_str = columns_gen_sql(
                        columns_list=columns_list, with_value_str=False)
                else:
                    columns_str = '*'

                query = 'SELECT {} FROM [{}].[{}]'.format(
                    columns_str, schema, table)

                if n_top_rows:
                    query = 'SELECT TOP ({}) * FROM [{}].[{}]'.format(
                        n_top_rows, schema, table)

                if type(where_clause) == str:
                    query += '\n' + where_clause

                if type(order_by) == str:
                    query += '\n' + 'ORDER BY {}'.format(order_by)

                if descending:
                    query += ' ' + 'DESC'

            else:
                if not columns_list:
                    raise Exception('Please Input Columns List')
                else:
                    columns_str = columns_gen_sql(
                        columns_list=columns_list, with_value_str=False)

                query = 'SELECT TOP ({}) {} \nFROM [{}].[{}]'.format(
                        n_top_rows, columns_str, schema, table)

                if type(where_clause) == str:
                    query += '\n' + where_clause

                if type(order_by) == str:
                    query += '\n' + 'ORDER BY {}'.format(order_by)

                if descending:
                    query += ' ' + 'DESC'

        df = pd.read_sql(query, self.cnxn)

        # converting data type sql to data type pandas
        col_info_df = self.return_sql_table_columns_info_df(
            table=table, schema=schema)
        col_list = list(df.columns)
        # Check if table exists in SQL metadata
        if table is None or table not in col_info_df["table_name"].unique():
            # If table is not exists in SQL metadata
            # Delete None & NaN in table (create null)
            df = df.fillna('')  # Make sure to assign the result back
            # Change datetime to string
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):  # More robust datetime check
                    df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            return df

        col_dtypes = [dtype_sql_to_pd(col_info_df.type_name[col_info_df.column_name == col].values[0]) for col in col_list]
        type_dict = {col_list[i]: col_dtypes[i] for i in range(len(col_list))}

        for col, col_dtype in type_dict.items():
            if col_dtype == "datetime64[ns]":
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif col_dtype == "int64":
                df[col] = pd.to_numeric(df[col], errors="coerce")
                try:
                    df[col] = df[col].astype("int64")
                except:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = df[col].astype(col_dtype)
        
        
        # col_dtypes = [
        #     dtype_sql_to_pd(
        #         col_info_df.type_name[col_info_df.column_name == col].values[0])
        #     for col in col_list]
        # type_dict = {}  # col as keys, dtype as values
        # for i in range(len(col_list)):
        #     type_dict[col_list[i]] = col_dtypes[i]

        # for col, col_dtype in type_dict.items():
        #     if col_dtype == 'datetime64[ns]':
        #         df[col] = pd.to_datetime(df[col], errors='coerce')

        #     elif col_dtype == 'int64':
        #         df[col] = pd.to_numeric(df[col], errors='coerce')
        #         try:
        #             # i have to put this here, because 1/0 becomes bool
        #             df[col] = df[col].astype('int64')
        #         except:
        #             # just in case there are NULL values
        #             df[col] = pd.to_numeric(df[col], errors='coerce')

        #     else:
        #         df[col] = df[col].astype(col_dtype)

        # df = df.astype(type_dict, errors='raise')
        if verbose:
            logging.info('Returning dataframe from query: \n{}'.format(query))
        return df

    def insert_df_to_table(self, table: str, df: pd.core.frame.DataFrame, schema: str = 'dbo', chunksize=None, method=None) -> str:
        """
        This function:
        1. takes a pandas dataframe
        2. checks if the dataframe's columns' dtypes are the same with the SQL tables'
        3. converts the datarame's columns' dtypes to a compatible dtype of the SQL tables'
        4. inserts the dataframe data into the SQL Table

        Args:
            table (str): [the SQL Table name]
            df (pd.core.frame.DataFrame): [the pandas dataframe]
            schema (str, optional): [the schema of the SQL Table]. Defaults to 'dbo'.
            chunksize(int, optional): Specify the number of rows in each batch to be written at a time. By default, all rows will be written at once.
           
            method ({None, ‘multi’, callable}, optional) Controls the SQL insertion clause used:

                - None : Uses standard SQL INSERT clause (one per row).
                - ‘multi’: Pass multiple values in a single INSERT clause.
                - callable with signature (pd_table, conn, keys, data_iter).




        Raises:
            Exception: [prints the error of insertion]

        Returns:
            str: [a confimation of the insertion]
        """
        logging.info('Validating df columns')
        self.validate_df_columns(df=df, schema=schema, table=table)
        logging.info('inserting data to {}.{}'.format(schema, table))
        df.to_sql(name=table,
                  schema=schema,
                  con=self.engine,
                  if_exists='append',
                  index=False,
                  chunksize=chunksize,
                  method=method
                  )
        logging.info('inserted {} row(s) to {}.{}'.format(len(df), schema, table))

    def truncate_all_rows_in_sql(self, table: str, schema: str = 'dbo'):
        """
        !!! This function !!!:
        1. Delete all rows in selected table

        Args:
            table (str): [description]
            schema (str, optional): [description]. Defaults to 'dbo'.

        Raises:
            Exception: [description]

        BE CAREFUL, ONCE DELETED CANNOT BE RECOVERED !!!
        """
        cursor = self.cnxn.cursor()
        is_error = False

        query = 'TRUNCATE TABLE [{}].[{}]'.format(schema, table)
        logging.info(query)
        try:
            cursor.execute(query)
        except Exception as e:
            logging.info('Error: ' + str(e))
            is_error = True

        self.cnxn.commit()
        cursor.close()
        if is_error:
            raise Exception('there is an error')
        else:
            logging.info('Successfully deleted all rows in [{}].[{}]!'.format(
                schema, table))

    def delete_row(self, table: str, where_clause: str, schema: str = 'dbo'):
        """
        !!! This function !!!:
        1. Delete selected row(s) in selected table

        Args:
            table (str): [description]
            where_clause (str): [description]
            schema (str, optional): [description]. Defaults to 'dbo'.

        Raises:
            Exception: [description]

        BE CAREFUL, ONCE DELETED CANNOT BE RECOVERED !!!
        """
        cursor = self.cnxn.cursor()
        is_error = False

        query = 'DELETE [{}].[{}]'.format(schema, table)

        query += '\n' + where_clause

        logging.info(query)
        try:
            cursor.execute(query)
        except Exception as e:
            logging.info('Error: ' + e)
            is_error = True

        self.cnxn.commit()
        cursor.close()
        if is_error:
            raise Exception('there is an error')
        else:
            logging.info('Successfully deleted selected row in [{}].[{}]'.format(
                schema, table))

    def update_row_sql(self, table: str, pd_series, where_clause: str, schema: str = 'dbo', verbose=False):
        """Updates a row in a an sql table

        Args:
            table (str): The target table
            pd_series ([type]): The row content to be inserted as update, make sure that the columns are the same
            where_clause (str): the where clause
            schema (str, optional): the target schema. Defaults to 'dbo'.
            verbose (bool, optional): if you want to logging.info information. Defaults to False.

        Raises:
            Exception: [description]
        """
        cursor = self.cnxn.cursor()
        is_error = False

        query = return_update_string(
            table=table, pd_series=pd_series, where_clause=where_clause, schema=schema)
        if verbose:
            logging.info(query)
        try:
            cursor.execute(query)
        except Exception as e:
            logging.info(e)
            is_error = True

        self.cnxn.commit()
        cursor.close()
        if is_error:
            raise Exception('there is an error')
        else:
            if verbose:
                logging.info('Successfully updated to selected row in [{}].[{}]'.format(
                    schema, table))

    def execute_query(self, query, verbose=False):
        cursor = self.cnxn.cursor()
        is_error = False
        if verbose:
            logging.info(query)
        try:
            cursor.execute(query)
        except Exception as e:
            logging.info(e)
            is_error = True

        self.cnxn.commit()
        cursor.close()
        if is_error:
            raise Exception('there is an error')
        else:
            if verbose:
                logging.info('executed query successfully')

    def update_multiple_rows_sql(self, table: str, df, where_clause: str, schema: str = 'dbo', verbose=False):
        """Updates a row in a an sql table

        Args:
            table (str): The target table
            df (pandas dataframe): The dataframe that is used for content update in the target sql Table, make sure all the columns are correct
            where_clause (str): the where clause
            schema (str, optional): the target schema. Defaults to 'dbo'.
            verbose (bool, optional): if you want to logging.info information. Defaults to False.

        Raises:
            Exception: [description]
        """

        df_list = []
        index_list = self.range_list_creator(n=len(df), gap=100)
        for pair in index_list:
            df_list.append(df[pair[0]:pair[1]])

        for df_inner in df_list:
            cursor = self.cnxn.cursor()
            is_error = False
            query = ""
            for i in range(len(df_inner)):
                ps = df_inner.iloc[i]
                query += return_update_string(
                    table=table, pd_series=ps, where_clause=where_clause, schema=schema)
            if verbose:
                logging.info(query)
            try:
                cursor.execute(query)
            except Exception as e:
                logging.info(e)
                is_error = True

            self.cnxn.commit()
            cursor.close()
            if is_error:
                raise Exception('there is an error')
            else:
                if verbose:
                    logging.info('Successfully updated to [{}].[{}]'.format(
                        schema, table))

    def range_list_creator(self, n, gap=100):
        """creates a list of list, with a certain range"""
        floor = 0
        ceil = 0
        ls = []
        for i in range(n):
            if i == (n-1):
                ceil = n
                ls.append([floor, ceil])
            else:
                k = i+1
                if ((k) % gap == 0):
                    ceil = k
                    ls.append([floor, ceil])
                    floor = ceil
        return ls

    def update_one_cell(self, table, column_name, value, where_clause, schema='dbo', verbose=False):
        """ use this if you only update one cell in a table"""
        query = f'UPDATE {schema}.{table} SET {column_name} = {value} {where_clause}'
        self.execute_query(query=query, verbose=verbose)

    def insert_one_row(self, table, values_dict, schema='dbo', verbose=False):
        """ use this if you only want to insert one row in the table
        the values_dict format is
        val_dict = {'column1': val1, 'column2': val2}"""

        columns_string = ''
        values_string = ''
        for k, v in values_dict.items():
            columns_string += f'{k}, '
            if type(v) == str:
                v = f"\'{v}\'"
            values_string += f'{v}, '
        columns_string = columns_string[:-2]
        values_string = values_string[:-2]

        query = f'INSERT INTO {schema}.{table} ({columns_string})\nVALUES ({values_string})'
        return query
        # self.execute_query(query, verbose)



def round_seconds_datetime64_ns(item):

    date = item.split()[0]
    h, m, s = [item.split()[1].split(':')[0],
               item.split()[1].split(':')[1],
               str(round(float(item.split()[1].split(':')[-1])))]
    result = (date + ' ' + h + ':' + m + ':' + s)
    if result[-3:] == ':60':
        result = result.replace(':60', ':59')

    return result
