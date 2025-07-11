# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a Python monorepo for DLM internal utilities, using Poetry workspaces to manage multiple packages. The repository is structured to house various internal tools, with CSP (Connect SQL with Pandas) being the primary package.

## Common Commands

### Development Setup
```bash
# Install Poetry if not already installed
pip install poetry

# Install dependencies for a specific package
cd packages/csp
poetry install

# Run tests for CSP package
cd packages/csp
python -m pytest tests/
```

### Testing
```bash
# Run all tests in CSP package
cd packages/csp
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_connection.py -v

# Run single test
python -m pytest tests/test_csp.py::test_function_name -v
```

### Package Management
```bash
# Build CSP package
cd packages/csp
poetry build

# Update version in packages/csp/pyproject.toml before release
# Version follows semantic versioning (MAJOR.MINOR.PATCH)
```

## Architecture

### Monorepo Structure
- Uses Poetry workspaces defined in root `pyproject.toml`
- Each package in `packages/` has its own `pyproject.toml`
- Packages are independently versioned and can be installed separately

### CSP Package Architecture
The CSP package provides a simplified interface for database connections:

1. **Connection Management** (`get_connection.py`):
   - Manages connection profiles stored in a global registry
   - Built-in profile: `azure_cico` 
   - Supports dynamic addition of custom profiles
   - Environment variables loaded from `.env` files

2. **Database Interface** (`connect_sql_pandas.py`):
   - `ConnectSqlPandas` class wraps SQLAlchemy connections
   - Automatic retry logic with exponential backoff
   - Returns query results as Pandas DataFrames
   - Supports CRUD operations and raw SQL execution

3. **Type Mapping**:
   - Automatic conversion between pandas dtypes and SQL types
   - Handles special characters in passwords via URL encoding

### Connection Flow
1. User calls `connect_to_database(profile_name)`
2. System retrieves connection params from profile registry
3. Creates SQLAlchemy engine with retry logic
4. Returns `ConnectSqlPandas` instance for querying

### Testing Strategy
- Integration tests that connect to actual databases
- Tests cover full CRUD operations
- Connection retry mechanism testing
- Environment variable and profile management testing