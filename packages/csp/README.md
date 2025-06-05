# 📦 CSP - Connect SQL with Pandas

A lightweight internal Python package to easily connect SQL databases and query data as Pandas DataFrames.

---

## 🚀 Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/DLM-Consultancy/dlm-internal-utils.git#subdirectory=packages/csp
```

### Updating Packages

To update to the latest version of a package, use the `--force-reinstall` flag with pip:

```bash
pip install --force-reinstall git+https://github.com/DLM-Consultancy/dlm-internal-utils.git#subdirectory=packages/csp
```

You can also install a specific branch or commit by adding `@branch-name` or `@commit-hash` to the URL:

```bash
pip install --force-reinstall git+https://github.com/DLM-Consultancy/dlm-internal-utils.git@main#subdirectory=packages/csp
```

---

## 🔧 Setup

### Built-in Connection Profiles

To use built-in connection profiles (e.g. `azure_cico`), create a `.env` file in your project root using the CLI command:

```bash
csp-init-env
```

Then fill in your credentials like this:

```env
azure_cico_server=your_server
azure_cico_database=your_database
azure_cico_username=your_username
azure_cico_password=your_password
```

### Custom Connection Profiles

You can add your own connection profiles dynamically:

```python
from csp import add_connection_profiles
import os

# Add custom connection profiles
add_connection_profiles({
    "report_db": {
        "server": os.getenv('report_db_server'),
        "database": os.getenv('report_db_database'),
        "username": os.getenv('report_db_username'),
        "password": os.getenv('report_db_password')
    }
})
```

For these custom profiles, define the environment variables in your `.env` file:

```env
report_db_server=report.internal.net
report_db_database=reports
report_db_username=reportuser
report_db_password=secretpass123
```

### List Available Profiles

You can list all available connection profiles:

```python
from csp import list_connection_profiles

# Get all registered connection profiles
profiles = list_connection_profiles()
print(profiles)  # ['azure_cico', 'report_db', ...]
```

---

## 🧑‍💻 Usage

```python
from csp import connect_to_database

# Connect using built-in profile (requires .env)
conn = connect_to_database("azure_cico")

# Run query
df = conn.read("SELECT * FROM your_table")
```

The connection system includes automatic retry logic for handling temporary connection issues.

---

## 🧩 Features

- ✅ Connect to SQL Server, Azure, and more
- ✅ Retrieve query results as Pandas DataFrames
- ✅ Execute raw SQL queries or from .sql files
- ✅ Add your own connection profiles dynamically
- ✅ Connection retry logic for reliability
- ✅ Type compatibility mapping between pandas and SQL

---

## 📌 Requirements

- Python 3.11+
- pandas
- sqlalchemy
- pyodbc
- python-dotenv

All required dependencies will be installed automatically.

---

## 🧠 Notes

- ⚠️ Do **not commit** your `.env` file. Use `.env.example` as a safe reference.
- 🛠 This package is for internal use within **DLM Group** only.
