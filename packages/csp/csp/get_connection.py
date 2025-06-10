import os
import time
import pyodbc
from typing import Dict
from . import connect_sql_pandas as csp
from dotenv import load_dotenv
import time

# Load environment variables from .env if available
load_dotenv()

# Built-in connection profiles
_connection_profiles = {
    "azure_cico": {
        "server": os.getenv('azure_cico_server'),
        "database": os.getenv('azure_cico_database'),
        "username": os.getenv('azure_cico_username'),
        "password": os.getenv('azure_cico_password')
    }
}

def add_connection_profiles(profiles: Dict[str, Dict[str, str]]) -> None:
    """
    Register one or more SQL connection profiles to the internal registry.

    Each profile must be a dictionary containing the following keys:
    - "server": SQL server address
    - "database": target database name
    - "username": login username
    - "password": login password

    🔐 For security and best practice, store your credentials in a `.env` file or in environment variables.
    You can refer to the provided `.env.example` for the expected variable names.

    ✅ Built-in available profiles:
    - azure_cico

    📌 Example usage:
        add_connection_profiles({
            "local": {
                "server": os.getenv('local_server'),
                "database": os.getenv('local_database'),
                "username": os.getenv('local_username'),
                "password": os.getenv('local_password'),
            },
            "azure": {
                "server": os.getenv('azure_server'),
                "database": os.getenv('azure_database'),
                "username": os.getenv('azure_username'),
                "password": os.getenv('azure_password'),
            }
        })
    """
    REQUIRED_KEYS = {"server", "database", "username", "password"}

    for name, profile in profiles.items():
        if not isinstance(profile, dict):
            raise ValueError(f"Profile '{name}' must be a dictionary.")
        missing = REQUIRED_KEYS - profile.keys()
        if missing:
            raise ValueError(f"Profile '{name}' is missing: {', '.join(missing)}")
        _connection_profiles[name] = profile

def list_connection_profiles():
    """
    Returns a list of all available connection profile names.

    Example:
        >>> list_connection_profiles()
        ['azure', 'SAP_DLM', 'mydb', 'analytics']
    """
    return list(_connection_profiles.keys())

def connect_to_database(connection_profile_name: str, verbose=False):
    """
    Dynamically connect to a registered SQL database.

    Args:
        connection_profile_name: str 
            The name of the connection profile to use (e.g. 'azure_cico').

    Returns:
        Sql_pd_cnxn
            A connection wrapper object for running SQL queries with pandas support.
    """
    if connection_profile_name not in _connection_profiles:
        raise ValueError(f"Connection profile '{connection_profile_name}' not found.")

    creds = _connection_profiles[connection_profile_name]

    attempts = 3
    while attempts > 0:
        try:
            connection = csp.SQLPandasConnection(
                server=creds["server"],
                database=creds["database"],
                username=creds["username"],
                password=creds["password"],
                verbose=verbose
            )
            return connection
        except pyodbc.Error as e:
            print(f"[{connection_profile_name}] Connection failed: {e}")
            time.sleep(2)
            attempts -= 1
            if attempts == 0:
                raise Exception(f"Could not connect to '{connection_profile_name}' after multiple attempts.")