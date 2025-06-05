from .get_connection import (
    connect_to_database,
    add_connection_profiles,
    list_connection_profiles,
)

from .connect_sql_pandas import SQLPandasConnection

import shutil
import importlib.resources as res


def copy_env_example(target_path=".env.example"):
    """
    Copy the bundled .env.example file to the desired target path (default: .env).
    """
    with res.path(__package__, ".env.example") as src:
        shutil.copy(src, target_path)
        print(f"✅ .env copied from: {src} → {target_path}")


__all__ = [
    "connect_to_database",
    "add_connection_profiles",
    "list_connection_profiles",
    "SQLPandasConnection",
    "copy_env_example", 
]