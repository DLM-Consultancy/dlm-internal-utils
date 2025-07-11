import pandas as pd
import pyodbc
import datetime
import numpy as np
import urllib
import sqlalchemy as sa
import logging
import time
logging.basicConfig(level=logging.INFO)
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
    # Special case for object columns and datetime fields - allow string/object types to be compatible with datetime
    # since they can be converted by _apply_sql_dtypes_to_dataframe later
    if pandas_dtype == 'object' and sql_dtype.lower() in PANDAS_COMPATIBLE_SQL_TYPES['datetime64[ns]']:
        return True
        
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
    column_data_type = []
    
    for column, value in data.items():
        # Log the column name, value, and type for every field
        if pd.isna(value) or value is None:
            formatted_value = 'NULL'
        # If value is True or False convert it into 1 or 0
        elif isinstance(value, bool):
            formatted_value = int(value)
            
        elif isinstance(value, (int, np.integer)):
            formatted_value = str(int(value))
        elif isinstance(value, (float, np.floating)):
            # Check if it's a whole number (like 630694.0)
            if value.is_integer():
                formatted_value = str(int(value))  # Remove decimal point for whole numbers
            else:
                formatted_value = str(value)  # Keep decimal for actual fractional numbers
        
        elif isinstance(value, str):
            # Check if it's possibly a datetime string
            if re.match(r'\d{4}-\d{2}-\d{2}.*\d{2}:\d{2}:\d{2}', value):
                # Handle potential datetime string with microseconds
                if '.' in value:
                    base, micros = value.split('.')
                    value = f"{base}.{micros[:3]}" if len(micros) > 3 else f"{base}.{micros}"
                formatted_value = f"'{value}'"
            # Check if it's a boolean string
            elif value.lower() in ['true', 'false']:
                formatted_value = int(value.lower() == 'true')
            # Check if it's a numeric string first
            elif value.replace('.', '', 1).isdigit():  # Fast check if it's numeric (allows one decimal point)
                # Check for leading zeros - if present, treat as string
                if value.startswith('0') and len(value) > 1 and value[1] != '.':
                    # Has leading zeros, keep as string
                    formatted_value = f"'{value}'"
                else:
                    try:
                        # Try to convert to float
                        num_val = float(value)
                        # If it's a whole number, convert to int to remove the decimal point
                        if num_val.is_integer():
                            formatted_value = str(int(num_val))
                        else:
                            formatted_value = str(num_val)  # Keep as float string without quotes
                    except ValueError:
                        # Not a valid number, treat as regular string
                        formatted_value = f"'{value}'"
            else:
                formatted_value = f"'{value}'"
        elif isinstance(value, pd._libs.tslibs.timestamps.Timestamp):
            # Format with 3 decimal places max for SQL Server compatibility
            datetime_str = value.strftime('%Y-%m-%d %H:%M:%S.%f')
            # Truncate to milliseconds (3 decimal places)
            if '.' in datetime_str:
                base, micros = datetime_str.split('.')
                datetime_str = f"{base}.{micros[:3]}"
            formatted_value = f"'{datetime_str}'"    
        elif isinstance(value, datetime.datetime):
            if value.microsecond > 0:
                # Format with milliseconds (3 decimal places)
                datetime_str = value.strftime('%Y-%m-%d %H:%M:%S.%f')
                # Truncate to 3 decimal places
                if '.' in datetime_str:
                    base, micros = datetime_str.split('.')
                    datetime_str = f"{base}.{micros[:3]}"
            else:
                datetime_str = value.strftime('%Y-%m-%d %H:%M:%S')
            formatted_value = f"'{datetime_str}'"
        else:
            # Check if it might be a custom datetime-like object
            try:
                if hasattr(value, 'strftime'):
                    logging.warning(f"{column}: Found object with strftime method - trying to format as datetime")
                    datetime_str = value.strftime('%Y-%m-%d %H:%M:%S')
                    if hasattr(value, 'microsecond') and value.microsecond > 0:
                        ms_str = str(value.microsecond).zfill(6)[:3]
                        datetime_str = f"{datetime_str}.{ms_str}"
                    formatted_value = f"'{datetime_str}'"
                else:
                    formatted_value = str(value)
            except Exception as e:
                logging.warning(f"{column}: Error handling value ({type(value)}): {str(e)}")
                # Default fallback
                formatted_value = str(value)
        
        set_clauses.append(f"[{column}] = {formatted_value}")
        column_data_type.append(f"[{column}] = {type(value)}")
    
    set_clause = ', '.join(set_clauses)
    query = f"UPDATE [{schema}].[{table}] SET {set_clause}"
    summary_column_data_type = ', '.join(column_data_type)
    
    if where_clause:
        query += f"\n{where_clause}"
    

    return query + ";", summary_column_data_type

def round_datetime_seconds(datetime_str: str) -> str:
    """Very simple solution - just remove microseconds completely"""
    if '.' in datetime_str:
        return datetime_str.split('.')[0]
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
                 pool_recycle: int = 1800,  # 30 minutes default
                 pool_pre_ping: bool = True,
                 max_retries: int = 3,
                 retry_interval: int = 2,
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
            pool_recycle: Recycle connections after this many seconds
            pool_pre_ping: Whether to test connection before use
            max_retries: Maximum number of connection retry attempts
            retry_interval: Seconds to wait between retries
            verbose: Enable verbose logging
        """
        self.server = server
        self.database = database
        self.verbose = verbose
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        
        # Build connection strings - handling special characters in password
        # For ODBC connection strings, we need to handle special characters properly
        # PyODBC expects certain characters to be escaped or the connection wrapped properly
        
        # Method 1: Direct PyODBC connection with dictionary (most reliable for special chars)
        retry_count = 0
        last_exception = None
        
        while retry_count < self.max_retries:
            try:
                # PyODBC connection using keyword arguments - handles special chars better
                self.connection = pyodbc.connect(
                    driver=driver,
                    server=server,
                    database=database,
                    uid=username,
                    pwd=password,  # PyODBC handles special characters internally
                    timeout=timeout,
                    trustservercertificate='yes'
                )
                
                # For SQLAlchemy - create engine using a creator function
                # This avoids connection string parsing issues with special characters
                def creator():
                    return pyodbc.connect(
                        driver=driver,
                        server=server,
                        database=database,
                        uid=username,
                        pwd=password,
                        timeout=timeout,
                        trustservercertificate='yes'
                    )
                
                # SQLAlchemy engine with connection pooling settings
                self.engine = sa.create_engine(
                    "mssql+pyodbc://",
                    creator=creator,
                    pool_pre_ping=pool_pre_ping,  # Test connection before use to avoid stale connections
                    pool_recycle=pool_recycle,    # Recycle connections after specified seconds
                    pool_timeout=timeout,         # How long to wait on a busy pool
                    pool_size=5,                  # Maintain up to 5 connections
                    max_overflow=10,              # Allow up to 10 additional connections
                    connect_args={
                        "connect_timeout": timeout
                    }
                )
                
                if verbose:
                    logging.info(f'Successfully connected to {server}/{database}')
                    # Debug: Show connection info with masked password
                    masked_password = '*' * min(len(password), 8) if password else ''
                    logging.debug(f'Connection info: server={server}, database={database}, username={username}, password={masked_password}')
                    
                return  # Connection successful, exit the retry loop
                    
            except Exception as e:
                retry_count += 1
                last_exception = e
                
                if verbose:
                    logging.warning(f'Connection attempt {retry_count} failed: {e}')
                    
                if retry_count < self.max_retries:
                    logging.info(f'Retrying in {self.retry_interval} seconds...')
                    time.sleep(self.retry_interval)
        
        # If we've exhausted all retries
        logging.error(f'Failed to connect to database after {self.max_retries} attempts. Last error: {last_exception}')
        raise last_exception
    

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
        # Create an explicit copy to avoid SettingWithCopyWarning
        result_df = df.copy()
        
        # Get table schema information
        table_info = self.get_table_info(table, schema)
        
        # If table info is empty, return DataFrame as-is with basic cleaning
        if table_info.empty:
            result_df = result_df.fillna('')
            # Convert datetime columns to string for safety
            for col in result_df.columns:
                if pd.api.types.is_datetime64_any_dtype(result_df[col]):
                    result_df.loc[:, col] = result_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            return result_df
        
        # Create mapping of column names to SQL types
        sql_types = dict(zip(table_info['column_name'], table_info['type_name']))
        
        # Apply correct data types
        for column in result_df.columns:
            if column in sql_types:
                sql_dtype = sql_types[column]
                pandas_dtype = sql_type_to_pandas(sql_dtype)
                
                try:
                    if pandas_dtype == 'datetime64[ns]':
                        # Identify dates outside the pandas datetime64[ns] supported range (1677-2262)
                        # We'll keep these as original values
                        preserved_dates = {}
                        
                        for idx in result_df.index:
                            val = result_df.loc[idx, column]
                            
                            # Handle string dates
                            if isinstance(val, str) and val:
                                try:
                                    year_str = val.split('-')[0]
                                    year = int(year_str)
                                    # Only convert to datetime if within pandas supported range
                                    if year < 1677 or year > 2262:
                                        preserved_dates[idx] = val
                                        if self.verbose:
                                            logging.warning(f"Date outside pandas supported range: {val}. Keeping as original value.")
                                except Exception:
                                    pass  # Not a date string format or couldn't parse year
                                    
                            # Handle Python datetime objects
                            elif hasattr(val, 'year'): 
                                try:
                                    # Only convert to datetime if within pandas supported range
                                    if val.year < 1677 or val.year > 2262:
                                        date_str = val.strftime('%Y-%m-%d %H:%M:%S')
                                        preserved_dates[idx] = date_str
                                        if self.verbose:
                                            logging.warning(f"Date outside pandas supported range: {date_str}. Keeping as original value.")
                                except Exception as e:
                                    logging.error(f"Error handling datetime: {e}")
                        
                        # Use pandas to convert what it can
                        result_df[column] = pd.to_datetime(
                            result_df[column], 
                            errors='coerce',
                            format='%Y-%m-%d %H:%M:%S'
                            )
                        
                        # Restore original values for dates outside pandas range
                        for idx, val in preserved_dates.items():
                            result_df.loc[idx, column] = val
                        
                        if preserved_dates and self.verbose:
                            logging.info(f"Preserved {len(preserved_dates)} dates outside pandas range for column '{column}'")
                            logging.info(f"Column dtype after preservation: {result_df[column].dtype}")
                            logging.info(f"Example values: {result_df[column].head()}")

                        
                    elif pandas_dtype in ['int64', 'Int64']:
                        # Create a completely new Series to avoid type compatibility issues
                        numeric_values = pd.to_numeric(result_df[column], errors='coerce')
                        # Make a copy with explicit index to ensure alignment
                        result_df = result_df.copy()
                        # Replace the column completely instead of using .loc
                        result_df[column] = pd.Series(numeric_values, index=result_df.index).astype('Int64')
                    else:
                        # Create a new Series to avoid type compatibility warnings
                        result_df = result_df.copy()
                        result_df[column] = pd.Series(result_df[column], index=result_df.index).astype(pandas_dtype)
                        
                except Exception as e:
                    if self.verbose:
                        logging.warning(f"Could not convert column '{column}' to {pandas_dtype}: {e}")
        
        return result_df
    
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
        else:
            # Change datetime to string
            for col in df.columns:
                if df[col].dtype == "datetime64[ns]":
                    df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            return df

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
    
    def execute_query(self, query: str, verbose: bool = False) -> int:
        """
        Execute a SQL query and return the number of affected rows.
        
        Args:
            query: SQL query to execute
            verbose: Enable verbose logging
            
        Returns:
            int: The number of rows affected by the query
            
        Raises:
            Exception: If the query execution fails
        """
        with self.get_cursor() as cursor:
            if verbose or self.verbose:
                logging.info(f'Executing query: {query}')
            try:
                cursor.execute(query)
                rows_affected = cursor.rowcount
                self.connection.commit()
                if verbose or self.verbose:
                    logging.info('Query executed successfully')
                return rows_affected
            except Exception as e:
                logging.error(e)
                raise

    
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
        query, summary_column_data_type = build_update_query(table, data, where_clause, schema)
        try:
            self.execute_query(query, verbose)        
        except Exception as e:
            logging.error(f"{e}\nColumn data types: {summary_column_data_type}")
            # Re-raise with additional context
            raise Exception(f"{e} - Column data types: {summary_column_data_type}") from e
        if verbose or self.verbose:
            logging.info(f'Successfully updated row in {schema}.{table}')
    
    def update_multiple_rows_sql(self, 
                              table: str, 
                              data_list: List[pd.Series], 
                              where_clauses: Union[List[str], str], 
                              schema: str = 'dbo',
                              verbose: bool = False) -> None:
        """
        Update multiple rows in SQL table (non-parameterized version for consistency with update_row_sql).
        
        Args:
            table: Target table name
            data_list: List of Series with column-value pairs (one per row)
            where_clauses: Either a list of WHERE clauses (one per row) or a single WHERE clause to apply to all rows
            schema: Schema name
            verbose: Enable verbose logging
        """
        if isinstance(where_clauses, str):
            # If where_clauses is a string, create a list with the same where clause for each data item
            where_clauses_list = [where_clauses] * len(data_list)
        else:
            # If where_clauses is already a list, make sure its length matches data_list
            if len(data_list) != len(where_clauses):
                raise ValueError("Length of data_list and where_clauses must match when where_clauses is a list")
            where_clauses_list = where_clauses

        for data, where_clause in zip(data_list, where_clauses_list):
            query, summary_column_data_type = build_update_query(table, data, where_clause, schema)
            try:
                self.execute_query(query, verbose)
            except Exception as e:
                logging.error(f"{e}\nColumn data types: {summary_column_data_type}")
                # Re-raise with additional context
                raise Exception(f"{e} - Column data types: {summary_column_data_type}") from e
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