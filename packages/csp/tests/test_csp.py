import os
import pandas as pd
import pytest
import datetime
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
            score INT,
            created_date DATETIME,
            last_update DATETIME
        )
        """
        conn.execute_query(create_query)

        # --- 2. INSERT ---
        current_time = datetime.datetime.now()
        yesterday = current_time - datetime.timedelta(days=1)
        df = pd.DataFrame([
            {"id": 1, "name": "Alice", "score": 90, "created_date": yesterday, "last_update": yesterday},
            {"id": 2, "name": "Bob", "score": 85, "created_date": yesterday, "last_update": yesterday}
        ])
        conn.insert_df_to_table(table=TEST_TABLE, df=df, schema=TEST_SCHEMA)

        # --- 3. GET ---
        result_df = conn.get_df(table=TEST_TABLE, schema=TEST_SCHEMA, where_clause="WHERE id = 1")
        assert not result_df.empty
        assert result_df.iloc[0]["name"] == "Alice"

        # --- 4. UPDATE single row ---
        current_time = datetime.datetime.now()
        update_row = pd.Series({"name": "Alice Updated", "score": 95, "last_update": current_time})
        conn.update_row_sql(table=TEST_TABLE, data=update_row, where_clause="WHERE id = 1", schema=TEST_SCHEMA)

        check_df = conn.get_df(table=TEST_TABLE, schema=TEST_SCHEMA, where_clause="WHERE id = 1")
        assert check_df.iloc[0]["name"] == "Alice Updated"
        # Verify datetime was properly updated 
        assert check_df.iloc[0]["last_update"].date() == current_time.date()

        # --- 5. UPDATE multiple rows ---
        update_time = datetime.datetime.now()
        data_list = [
            pd.Series({"name": "Bob Updated", "score": 99, "last_update": update_time}),
            pd.Series({"name": "Alice Final", "score": 100, "last_update": update_time})
        ]
        where_clauses = ["WHERE id = 2", "WHERE id = 1"]
        conn.update_multiple_rows_sql(table=TEST_TABLE, data_list=data_list, where_clauses=where_clauses, schema=TEST_SCHEMA)

        multi_df = conn.get_df(table=TEST_TABLE, schema=TEST_SCHEMA, where_clause="WHERE id IN (1, 2)")
        assert "Bob Updated" in multi_df["name"].values
        assert "Alice Final" in multi_df["name"].values
        # Verify datetime was properly updated for both rows
        for i in range(len(multi_df)):
            assert multi_df.iloc[i]["last_update"].date() == update_time.date()

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

def test_string_dates_to_datetime():
    """
    Test that string dates in object columns can be inserted into datetime SQL columns.
    This specifically tests the fix for datetime validation allowing object columns to be
    compatible with SQL datetime fields.
    """
    conn = SQLPandasConnection(
        server=os.getenv("azure_cico_server"),
        database=os.getenv("azure_cico_database"),
        username=os.getenv("azure_cico_username"),
        password=os.getenv("azure_cico_password"),
        verbose=True
    )

    TEST_TABLE_DATES = "test_table_dates"
    TEST_SCHEMA = "cico"

    try:
        # --- 1. SETUP: Create table with datetime columns ---
        create_query = f"""
        CREATE TABLE [{TEST_SCHEMA}].[{TEST_TABLE_DATES}] (
            id INT PRIMARY KEY,
            name NVARCHAR(100),
            created_date DATETIME,
            end_date DATETIME NULL
        )
        """
        conn.execute_query(create_query)

        # --- 2. INSERT: Using string dates (object dtype) ---
        # IMPORTANT: These are strings, not datetime objects
        df = pd.DataFrame([
            {"id": 1, "name": "Test1", "created_date": "2025-01-01 10:30:00", "end_date": "2025-12-31 23:59:59"},
            {"id": 2, "name": "Test2", "created_date": "2025-06-15 08:45:00", "end_date": None}
        ])
        
        # Confirm these are actually object dtype, not datetime
        assert df["created_date"].dtype == 'object'
        assert df["end_date"].dtype == 'object'
        
        # Insert the DataFrame with string dates into the table with datetime columns
        conn.insert_df_to_table(table=TEST_TABLE_DATES, df=df, schema=TEST_SCHEMA)

        # --- 3. GET and validate data ---
        result_df = conn.get_df(table=TEST_TABLE_DATES, schema=TEST_SCHEMA, order_by="id")

        # Verify data was inserted correctly
        assert result_df.shape[0] == 2
        
        # String dates should be converted to datetime objects
        assert isinstance(result_df.iloc[0]["created_date"], pd._libs.tslibs.timestamps.Timestamp)
        assert result_df.iloc[0]["created_date"].year == 2025
        assert result_df.iloc[0]["created_date"].month == 1
        assert result_df.iloc[0]["created_date"].day == 1
        
        # Verify end_date NULL handling
        assert isinstance(result_df.iloc[0]["end_date"], pd._libs.tslibs.timestamps.Timestamp)
        assert pd.isna(result_df.iloc[1]["end_date"])

    finally:
        # --- 4. CLEANUP: Drop test table ---
        drop_query = f"DROP TABLE [{TEST_SCHEMA}].[{TEST_TABLE_DATES}]"
        conn.execute_query(drop_query)
        conn.close()