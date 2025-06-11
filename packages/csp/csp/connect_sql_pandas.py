import pandas as pd
import pyodbc
import datetime
import numpy as np
import urllib
import sqlalchemy as sa
import logging
from typing import Optional, List, Union, Any
from contextlib import contextmanager
from sqlalchemy.exc import IntegrityError
import re

# Complete type compatibility mapping - single source of truth
PANDAS_COMPATIBLE_SQL_TYPES = {
    'int64': {'int', 'bigint', 'smallint', 'tinyint', 'int identity'},
    'int32': {'int', 'bigint', 'smallint', 'tinyint'},
    'Int64': {'int', 'bigint', 'smallint', 'tinyint'},  # Nullable int
    'float64': {'float', 'decimal', 'numeric', 'real', 'money', 'smallmoney'},
    'float32': {'float', 'decimal', 'numeric', 'real'},
    'object': {'varchar', 'nvarchar', 'char', 'text', 'ntext'},
    'bool': {'bit', 'boolean'},
    'datetime64[ns]': {'datetime', 'datetime2', 'smalldatetime', 'date', 'time'}
}

def sql_type_to_pandas(sql_dtype: str) -> str:
    """
    Convert SQL data type to appropriate pandas data type.
    
    Args:
        sql_dtype: SQL data type string
        
    Returns:
        Corresponding pandas data type
    """
    sql_lower = sql_dtype.lower()
    
    # Find which pandas type is compatible with this SQL type
    for pandas_type, sql_types in PANDAS_COMPATIBLE_SQL_TYPES.items():
        if sql_lower in sql_types:
            return pandas_type
    
    return 'object'  # Default fallback

def are_types_compatible(pandas_dtype: str, sql_dtype: str) -> bool:
    """
    Check if pandas and SQL data types are compatible.
    
    Args:
        pandas_dtype: Pandas data type string
        sql_dtype: SQL data type string
        
    Returns:
        True if types are compatible
    """
    compatible_sql_types = PANDAS_COMPATIBLE_SQL_TYPES.get(pandas_dtype, set())
    return sql_dtype.lower() in compatible_sql_types

def pandas_dtype_to_python(value: Any) -> Any:
    """
    Convert pandas data types to Python native types.
    
    Args:
        value: Value with pandas data type
        
    Returns:
        Value converted to Python native type
    """
    if pd.isna(value):
        return None
        
    value_type = type(value)
    
    if value_type == np.int64:
        return int(value)
    elif value_type == np.float64:
        return float(value)
    elif value_type == np.datetime64:
        return numpy_datetime_to_python(value)
    elif value_type == str:
        return str(value)
    elif value_type == pd._libs.tslibs.timestamps.Timestamp:
        return value.to_pydatetime()
    else:
        return value

def numpy_datetime_to_python(dt64: np.datetime64) -> datetime.datetime:
    """
    Convert numpy datetime64[ns] to Python datetime.
    
    Args:
        dt64: numpy datetime64 object
        
    Returns:
        Python datetime object
    """
    timestamp = (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    return datetime.datetime.fromtimestamp(timestamp)

def build_columns_string(columns: List[str], include_placeholders: bool = False) -> Union[str, tuple]:
    """
    Build SQL columns string and optionally placeholders.
    
    Args:
        columns: List of column names
        include_placeholders: Whether to include SQL placeholders
        
    Returns:
        Column string or tuple of (columns_string, placeholders_string)
    """
    columns_str = ', '.join(columns)
    
    if include_placeholders:
        placeholders_str = ', '.join(['?'] * len(columns))
        return columns_str, placeholders_str
    
    return columns_str

def build_insert_query(table: str, columns: List[str], schema: str = 'dbo') -> str:
    """
    Build SQL INSERT query string.
    
    Args:
        table: Table name
        columns: List of column names
        schema: Schema name
        
    Returns:
        SQL INSERT query string
    """
    columns_str, placeholders_str = build_columns_string(columns, include_placeholders=True)
    return f"INSERT INTO [{schema}].[{table}] ({columns_str}) VALUES ({placeholders_str})"

def build_update_query(table: str, data: pd.Series, where_clause: str = "", schema: str = 'dbo') -> str:
    """
    Build SQL UPDATE query string.
    
    Args:
        table: Table name
        data: Series with column names as index and values as data
        where_clause: WHERE clause
        schema: Schema name
        
    Returns:
        SQL UPDATE query string
    """
    set_clauses = []
    
    for column, value in data.items():
        if pd.isna(value) or value is None:
            formatted_value = 'NULL'
        elif isinstance(value, str):
            formatted_value = f"'{value}'"
        elif isinstance(value, pd._libs.tslibs.timestamps.Timestamp):
            datetime_str = round_datetime_seconds(str(value))
            formatted_value = f"'{datetime_str}'"
        else:
            formatted_value = str(value)
        
        set_clauses.append(f"[{column}] = {formatted_value}")
    
    set_clause = ', '.join(set_clauses)
    query = f"UPDATE [{schema}].[{table}] SET {set_clause}"
    
    if where_clause:
        query += f"\n{where_clause}"
    
    return query + ";"

def round_datetime_seconds(datetime_str: str) -> str:
    """
    Round seconds in datetime string to avoid precision issues.
    
    Args:
        datetime_str: Datetime as string
        
    Returns:
        Datetime string with rounded seconds
    """
    try:
        parts = datetime_str.split()
        if len(parts) < 2:
            return datetime_str
            
        date_part = parts[0]
        time_parts = parts[1].split(':')
        
        if len(time_parts) >= 3:
            hour, minute = time_parts[0], time_parts[1]
            second = str(round(float(time_parts[2])))
            
            # Handle edge case where rounding gives 60 seconds
            if second == '60':
                second = '59'
            
            return f"{date_part} {hour}:{minute}:{second}"
    except (ValueError, IndexError):
        pass
    
    return datetime_str

class SQLPandasConnection:
    """
    A class for managing connections and operations between Python/Pandas and SQL Server.
    
    Main Methods:
        - get_dataframe: Retrieve data as DataFrame
        - get_table_info: Get table column information
        - insert_dataframe: Insert DataFrame to table
        - update_row: Update single row
        - delete_rows: Delete rows with WHERE clause
    """

    def __init__(self, 
                 server: str, 
                 database: str, 
                 username: str, 
                 password: str,
                 driver: str = '{ODBC Driver 18 for SQL Server}',
                 timeout: int = 30,
                 verbose: bool = True):
        """
        Initialize SQL Server connection.
        
        Args:
            server: SQL Server instance
            database: Database name
            username: Username for authentication
            password: Password for authentication
            driver: ODBC driver to use
            timeout: Connection timeout in seconds
            verbose: Enable verbose logging
        """
        self.server = server
        self.database = database
        self.verbose = verbose
        
        # Build connection strings
        connection_params = (
            f'DRIVER={driver};'
            f'SERVER={server};'
            f'DATABASE={database};'
            f'UID={username};'
            f'PWD={password};'
            f'Connection Timeout={timeout};'
            'TrustServerCertificate=yes;'
        )
        
        encoded_params = urllib.parse.quote_plus(connection_params)
        
        try:
            # PyODBC connection for cursor operations
            self.connection = pyodbc.connect(connection_params)
            
            # SQLAlchemy engine for DataFrame operations
            self.engine = sa.create_engine(f"mssql+pyodbc:///?odbc_connect={encoded_params}")
            
            if verbose:
                logging.info(f'Successfully connected to {server}/{database}')
                
        except Exception as e:
            logging.error(f'Failed to connect to database: {e}')
            raise
    

    @contextmanager
    def get_cursor(self):
        """Context manager for database cursors."""
        cursor = self.connection.cursor()
        try:
            yield cursor
        except Exception as e:
            logging.error(f"Database operation failed: {e}")
            raise
        finally:
            cursor.close()
        
    def get_table_info(self, table: str, schema: str = 'dbo') -> pd.DataFrame:
        """
        Get table column information as DataFrame.
        
        Args:
            table: Table name
            schema: Schema name
            
        Returns:
            DataFrame with table column information
        """
        with self.get_cursor() as cursor:
            columns_info = cursor.columns(table=table, schema=schema)
            
            table_info = []
            for column in columns_info:
                table_info.append([
                    column.table_cat, column.table_name, column.column_name,
                    column.type_name, column.column_size, column.buffer_length,
                    column.decimal_digits, column.num_prec_radix, column.nullable,
                    column.remarks, column.column_def, column.sql_data_type,
                    column.sql_datetime_sub, column.char_octet_length,
                    column.ordinal_position, column.is_nullable
                ])
        
        columns = [
            'table_cat', 'table_name', 'column_name', 'type_name', 'column_size',
            'buffer_length', 'decimal_digits', 'num_prec_radix', 'nullable',
            'remarks', 'column_def', 'sql_data_type', 'sql_datetime_sub',
            'char_octet_length', 'ordinal_position', 'is_nullable'
        ]
        
        return pd.DataFrame(table_info, columns=columns)
    
    def _validate_dataframe_dtypes(self, df: pd.DataFrame, table: str, schema: str) -> None:
        """
        Validate DataFrame column data types against SQL table schema.
        
        Args:
            df: DataFrame to validate
            table: Target table name
            schema: Schema name
        """
        table_info = self.get_table_info(table, schema)
        
        # Create mapping of column names to SQL types
        sql_types = dict(zip(table_info['column_name'], table_info['type_name']))
        
        for column in df.columns:
            if column not in sql_types:
                raise ValueError(f"Column '{column}' not found in table {schema}.{table}")
            
            df_dtype = str(df[column].dtype)
            sql_dtype = sql_types[column]
            
            # Use compatibility check instead of exact mapping
            if not are_types_compatible(df_dtype, sql_dtype):
                raise ValueError(
                    f"Data type mismatch for column '{column}': "
                    f"DataFrame has {df_dtype}, SQL table expects {sql_dtype}"
                )
        
        if self.verbose:
            logging.info("DataFrame column types validated successfully")
    
    def _apply_sql_dtypes_to_dataframe(self, df: pd.DataFrame, table: str, schema: str) -> pd.DataFrame:
        """
        Apply SQL table data types to DataFrame columns.
        
        Args:
            df: Input DataFrame
            table: SQL table name
            schema: Schema name
            
        Returns:
            DataFrame with corrected data types
        """
        # Get table schema information
        table_info = self.get_table_info(table, schema)
        
        # If table info is empty, return DataFrame as-is with basic cleaning
        if table_info.empty:
            df = df.fillna('')
            # Convert datetime columns to string for safety
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            return df
        
        # Create mapping of column names to SQL types
        sql_types = dict(zip(table_info['column_name'], table_info['type_name']))
        
        # Apply correct data types
        for column in df.columns:
            if column in sql_types:
                sql_dtype = sql_types[column]
                pandas_dtype = sql_type_to_pandas(sql_dtype)
                
                try:
                    if pandas_dtype == 'datetime64[ns]':
                        df[column] = pd.to_datetime(df[column], errors='coerce')
                    elif pandas_dtype in ['int64', 'Int64']:
                        df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
                    else:
                        df[column] = df[column].astype(pandas_dtype)
                        
                except Exception as e:
                    if self.verbose:
                        logging.warning(f"Could not convert column '{column}' to {pandas_dtype}: {e}")
        
        return df
    
    def get_df(self, 
            table: Optional[str] = None,
            schema: str = 'dbo',
            columns: Optional[List[str]] = None,
            where_clause: Optional[str] = None,
            order_by: Optional[str] = None,
            limit: Optional[int] = None,
            descending: bool = False,
            query: Optional[str] = None,
            verbose: bool = False) -> pd.DataFrame:
        """
        Retrieve data from SQL table as pandas DataFrame.

        Args:
        table: Table name
        schema: Schema name
        columns: List of columns to select (None for SELECT all)
        where_clause: WHERE clause
        order_by: ORDER BY column
        limit: Number of rows to limit (TOP clause)
        descending: Sort in descending order
        query: Custom SQL query (overrides other parameters)
        verbose: Enable verbose logging

        Returns:
        pandas DataFrame with query results
        """
        if query:
            final_query = query
        else:
            if not table:
                raise ValueError("Table name is required when not providing custom query")

            # Build column selection
            if columns:
                columns_str = build_columns_string(columns)
            else:
                columns_str = '*'

            # Build base query
            if limit:
                final_query = f"SELECT TOP ({limit}) {columns_str} FROM [{schema}].[{table}]"
            else:
                final_query = f"SELECT {columns_str} FROM [{schema}].[{table}]"

            # Add WHERE clause
            if where_clause:
                final_query += f"\n{where_clause}"

            # Add ORDER BY clause
            if order_by:
                final_query += f"\nORDER BY {order_by}"
                if descending:
                    final_query += " DESC"

        # Execute query
        df = pd.read_sql(final_query, self.engine)

        # Apply SQL data types if table is specified
        if table:
            df = self._apply_sql_dtypes_to_dataframe(df, table, schema)

        if verbose:
            logging.info(f'Executed query: {final_query}')
            logging.info(f'Returned {len(df)} rows')

        return df
    
    def insert_df_to_table(self, 
                        table: str, 
                        df: pd.DataFrame, 
                        schema: str = 'dbo',
                        chunksize: Optional[int] = None,
                        method: Optional[str] = None,
                        validate_dtypes: bool = True) -> None:
        """
        Insert pandas DataFrame into SQL table.
        
        Args:
            table: Target table name
            df: DataFrame to insert
            schema: Schema name
            chunksize: Number of rows per batch
            method: Insertion method ('multi' for bulk insert)
            validate_dtypes: Whether to validate data types
        """
        df = self._apply_sql_dtypes_to_dataframe(df, table, schema)

        if validate_dtypes:
            self._validate_dataframe_dtypes(df, table, schema)
        
        if self.verbose:
            logging.info(f'Inserting {len(df)} rows to {schema}.{table}')
        
        try:
            df.to_sql(
                name=table,
                schema=schema,
                con=self.engine,
                if_exists='append',
                index=False,
                chunksize=chunksize,
                method=method
            )
            
            success_msg = f'Successfully inserted {len(df)} rows to {schema}.{table}'
            if self.verbose:
                logging.info(success_msg)
            
            return success_msg, None
            
        except IntegrityError as e:
            error_msg = str(e)
            logging.error(f"Integrity error occurred: {error_msg}")
            
            # Extract the problematic values from the error message
            # Example pattern: "The duplicate key value is (68, 2)"
            match = re.search(r"key value is \((.*?)\)", error_msg)
            if match:
                values = match.group(1).split(',')
                values = [v.strip() for v in values]
                
                # Extract column names from the constraint name
                constraint_match = re.search(r"constraint '([^']*)'", error_msg)
                if constraint_match:
                    constraint_name = constraint_match.group(1)
                    # Extract column names from UQ_TableName_Col1_Col2 format
                    cols = constraint_name.split('_')[2:-1]  # Skip UQ, TableName, and PlantID
                    
                    # Create a filter for the dataframe to find the problematic row
                    filter_conditions = []
                    for col, val in zip(cols, values):
                        try:
                            # Try to convert value to numeric if possible
                            numeric_val = pd.to_numeric(val)
                            filter_conditions.append(df[col] == numeric_val)
                        except:
                            filter_conditions.append(df[col] == val)
                    
                    # Combine all conditions
                    final_filter = filter_conditions[0]
                    for condition in filter_conditions[1:]:
                        final_filter = final_filter & condition
                    
                    problematic_data = df[final_filter].copy()
                    if not problematic_data.empty:
                        logging.error(f"Found problematic data:\n{problematic_data}")
                        return f"Error inserting data: {error_msg}", problematic_data
            
            # If we couldn't parse the error message, return the full dataframe
            return f"Error inserting data: {error_msg}", df
    
    def execute_query(self, query: str, verbose: bool = False) -> None:
        """
        Execute a SQL query without returning results.
        
        Args:
            query: SQL query to execute
            verbose: Enable verbose logging
        """
        with self.get_cursor() as cursor:
            if verbose or self.verbose:
                logging.info(f'Executing query: {query}')
            
            cursor.execute(query)
            self.connection.commit()
            
            if verbose or self.verbose:
                logging.info('Query executed successfully')
    
    def update_row_sql(self, 
                        table: str, 
                        data: pd.Series, 
                        where_clause: str,
                        schema: str = 'dbo',
                        verbose: bool = False) -> None:
        """
        Update a single row in SQL table.
        
        Args:
            table: Target table name
            data: Series with column-value pairs
            where_clause: WHERE clause for row identification
            schema: Schema name
            verbose: Enable verbose logging
        """
        query = build_update_query(table, data, where_clause, schema)
        self.execute_query(query, verbose)
        
        if verbose or self.verbose:
            logging.info(f'Successfully updated row in {schema}.{table}')
    
    def update_multiple_rows_sql(self, 
                              table: str, 
                              data_list: List[pd.Series], 
                              where_clauses: List[str], 
                              schema: str = 'dbo',
                              verbose: bool = False) -> None:
        """
        Update multiple rows in SQL table (non-parameterized version for consistency with update_row_sql).
        
        Args:
            table: Target table name
            data_list: List of Series with column-value pairs (one per row)
            where_clauses: List of WHERE clauses for row identification (one per row)
            schema: Schema name
            verbose: Enable verbose logging
        """
        if len(data_list) != len(where_clauses):
            raise ValueError("Length of data_list and where_clauses must match")

        for data, where_clause in zip(data_list, where_clauses):
            query = build_update_query(table, data, where_clause, schema)
            self.execute_query(query, verbose)

            if verbose or self.verbose:
                logging.info(f"Successfully updated row in {schema}.{table}")

    def delete_rows(self, 
                   table: str, 
                   where_clause: str, 
                   schema: str = 'dbo',
                   verbose: bool = False) -> None:
        """
        Delete rows from SQL table.
        
        Args:
            table: Target table name
            where_clause: WHERE clause for row identification
            schema: Schema name
            verbose: Enable verbose logging
        """
        query = f"DELETE FROM [{schema}].[{table}] {where_clause}"
        self.execute_query(query, verbose)
        
        if verbose or self.verbose:
            logging.info(f'Successfully deleted rows from {schema}.{table}')
    
    def truncate_table(self, table: str, schema: str = 'dbo') -> None:
        """
        Truncate all rows from table.
        
        Args:
            table: Target table name
            schema: Schema name
        """
        query = f"TRUNCATE TABLE [{schema}].[{table}]"
        self.execute_query(query, verbose=True)
        
        logging.info(f'Successfully truncated table {schema}.{table}')
    
    def update_cell(self, 
                table: str, 
                column: str, 
                value: Any, 
                where_clause: str,
                schema: str = 'dbo',
                verbose: bool = False) -> None:
        """
        Securely update a single cell using parameterized SQL.
        """
        query = f"UPDATE [{schema}].[{table}] SET [{column}] = ? {where_clause}"
        value = pandas_dtype_to_python(value)

        if verbose or self.verbose:
            logging.info(f"Executing update: {query} with value: {value}")

        with self.get_cursor() as cursor:
            cursor.execute(query, [value])
            self.connection.commit()

        if verbose or self.verbose:
            logging.info(f"Successfully updated [{schema}].[{table}] SET [{column}] = {value}")
    
    def close(self) -> None:
        """Close database connections."""
        try:
            if hasattr(self, 'connection'):
                self.connection.close()
            if hasattr(self, 'engine'):
                self.engine.dispose()
            
            if self.verbose:
                logging.info('Database connections closed')
        except Exception as e:
            logging.error(f'Error closing connections: {e}')