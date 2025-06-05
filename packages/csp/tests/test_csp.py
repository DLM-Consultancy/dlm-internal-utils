import os
import pandas as pd
import pytest
from dotenv import load_dotenv
from csp.connect_sql_pandas import SQLPandasConnection

load_dotenv()

TEST_TABLE = "test_table"
TEST_SCHEMA = "cico"

def test_sqlpandasconnection_full_flow():
    conn = SQLPandasConnection(
        server=os.getenv("azure_cico_server"),
        database=os.getenv("azure_cico_database"),
        username=os.getenv("azure_cico_username"),
        password=os.getenv("azure_cico_password"),
        verbose=True
    )

    try:
        # --- 1. SETUP: Create test table using execute_query ---
        create_query = f"""
        CREATE TABLE [{TEST_SCHEMA}].[{TEST_TABLE}] (
            id INT PRIMARY KEY,
            name NVARCHAR(100),
            score INT
        )
        """
        conn.execute_query(create_query)

        # --- 2. INSERT ---
        df = pd.DataFrame([
            {"id": 1, "name": "Alice", "score": 90},
            {"id": 2, "name": "Bob", "score": 85}
        ])
        conn.insert_df_to_table(table=TEST_TABLE, df=df, schema=TEST_SCHEMA)

        # --- 3. GET ---
        result_df = conn.get_df(table=TEST_TABLE, schema=TEST_SCHEMA, where_clause="WHERE id = 1")
        assert not result_df.empty
        assert result_df.iloc[0]["name"] == "Alice"

        # --- 4. UPDATE single row ---
        update_row = pd.Series({"name": "Alice Updated", "score": 95})
        conn.update_row_sql(table=TEST_TABLE, data=update_row, where_clause="WHERE id = 1", schema=TEST_SCHEMA)

        check_df = conn.get_df(table=TEST_TABLE, schema=TEST_SCHEMA, where_clause="WHERE id = 1")
        assert check_df.iloc[0]["name"] == "Alice Updated"

        # --- 5. UPDATE multiple rows ---
        data_list = [
            pd.Series({"name": "Bob Updated", "score": 99}),
            pd.Series({"name": "Alice Final", "score": 100})
        ]
        where_clauses = ["WHERE id = 2", "WHERE id = 1"]
        conn.update_multiple_rows_sql(table=TEST_TABLE, data_list=data_list, where_clauses=where_clauses, schema=TEST_SCHEMA)

        multi_df = conn.get_df(table=TEST_TABLE, schema=TEST_SCHEMA, where_clause="WHERE id IN (1, 2)")
        assert "Bob Updated" in multi_df["name"].values
        assert "Alice Final" in multi_df["name"].values

        # --- 6. DELETE rows ---
        conn.delete_rows(table=TEST_TABLE, where_clause="WHERE id IN (1, 2)", schema=TEST_SCHEMA)
        final_df = conn.get_df(table=TEST_TABLE, schema=TEST_SCHEMA, where_clause="WHERE id IN (1, 2)")
        assert final_df.empty

    finally:
        # --- 7. CLEANUP: Drop test table ---
        drop_query = f"DROP TABLE [{TEST_SCHEMA}].[{TEST_TABLE}]"
        conn.execute_query(drop_query)
        conn.close()

def test_sqlpandasconnection_with_null_values():
    
    conn = SQLPandasConnection(
        server=os.getenv("azure_cico_server"),
        database=os.getenv("azure_cico_database"),
        username=os.getenv("azure_cico_username"),
        password=os.getenv("azure_cico_password"),
        verbose=True
    )

    TEST_TABLE_NULL = "test_table_null"
    TEST_SCHEMA = "cico"

    try:
        # --- 1. SETUP: Create table ---
        create_query = f"""
        CREATE TABLE [{TEST_SCHEMA}].[{TEST_TABLE_NULL}] (
            id INT PRIMARY KEY,
            name NVARCHAR(100) NULL,
            score INT NULL
        )
        """
        conn.execute_query(create_query)

        # --- 2. INSERT: Including null values ---
        df = pd.DataFrame([
            {"id": 1, "name": "Alice", "score": 90},
            {"id": 2, "name": None, "score": 75},    # null name
            {"id": 3, "name": "Charlie", "score": None},  # null score
            {"id": 4, "name": None, "score": None}   # all nullable
        ])
        conn.insert_df_to_table(table=TEST_TABLE_NULL, df=df, schema=TEST_SCHEMA)

        # --- 3. GET: Validate data with nulls inserted properly ---
        result_df = conn.get_df(table=TEST_TABLE_NULL, schema=TEST_SCHEMA, order_by="id")
        print(result_df)

        assert result_df.shape[0] == 4
        assert pd.isna(result_df[result_df["id"] == 2]["name"].values[0])
        assert pd.isna(result_df[result_df["id"] == 3]["score"].values[0])
        assert pd.isna(result_df[result_df["id"] == 4]["name"].values[0])
        assert pd.isna(result_df[result_df["id"] == 4]["score"].values[0])

        # --- 4. CLEANUP rows ---
        conn.delete_rows(table=TEST_TABLE_NULL, where_clause="WHERE id IN (1, 2, 3, 4)", schema=TEST_SCHEMA)
        check_df = conn.get_df(table=TEST_TABLE_NULL, schema=TEST_SCHEMA)
        assert check_df.empty

    finally:
        # --- 5. CLEANUP: Drop test table ---
        drop_query = f"DROP TABLE [{TEST_SCHEMA}].[{TEST_TABLE_NULL}]"
        conn.execute_query(drop_query)
        conn.close()