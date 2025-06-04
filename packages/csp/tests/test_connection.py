import pytest
import os
from csp.get_connection import (
    list_connection_profiles,
    add_connection_profiles,
    connect_to_database
)

def test_add_and_list_profiles():
    test_profiles = {
        "mock_db": {
            "server": "localhost",
            "database": "mock_db",
            "username": "user",
            "password": "pass"
        }
    }

    # Add the test profile
    add_connection_profiles(test_profiles)

    # Check that it's listed
    available = list_connection_profiles()

    assert "mock_db" in available, "'mock_db' was not found in the list of connection profiles"

def test_connect_to_database_with_env(monkeypatch):
    # Mock environment variables for azure_cico
    profile_name = "azure_cico"

    # Assert env vars exist
    for var in ["azure_cico_server", "azure_cico_database", "azure_cico_username", "azure_cico_password"]:
        assert os.getenv(var), f"Missing required env var: {var}"

    try:
        conn = connect_to_database(profile_name)
        print(f"Connected: {conn}")
        assert conn is not None
    except Exception as e:
        pytest.fail(f"Connection to profile '{profile_name}' failed with error: {e}")
