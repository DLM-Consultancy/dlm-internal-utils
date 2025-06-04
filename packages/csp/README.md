# 📦 CSP - Connect SQL with Pandas

A lightweight internal Python package to easily connect SQL databases and query data as Pandas DataFrames.

---

## 🚀 Installation

Install directly from GitHub (private repo access required):

```bash
pip install git+https://github.com/DLM-Consultancy/dlm-internal-utils.git#subdirectory=packages/csp
```

---

## 🔧 Setup

To use built-in connection profiles (e.g. `azure_cico`), create a `.env` file in your project root or copy the provided template:

```bash
cp .env.example .env
```

Then fill in your credentials like this:

```env
azure_cico_server=dlmdbaz.database.windows.net
azure_cico_database=DB01
azure_cico_username=your_username
azure_cico_password=your_password
```

For new connection profiles using `os.getenv`, you may also define them in the `.env` file.  
Example for a custom connection named `report_db`:

```env
report_db_server=report.internal.net
report_db_database=reports
report_db_username=reportuser
report_db_password=secretpass123
```

Then access it via:

```python
conn = connect_to_database("report_db")
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

---

## 🧩 Features

- ✅ Connect to SQL Server, Azure, and more
- ✅ Retrieve query results as Pandas DataFrames
- ✅ Execute raw SQL queries or from .sql files
- ✅ Add your own connection profiles dynamically

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
