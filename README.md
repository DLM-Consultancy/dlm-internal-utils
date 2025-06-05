# DLM Internal Utilities

A collection of internal utility packages developed by DLM Consultancy for data processing, database connections, and other common tasks.

## 📦 Available Packages

### CSP (Connect SQL with Pandas)
A lightweight package to easily connect to SQL databases and query data as Pandas DataFrames.

```bash
pip install git+https://github.com/DLM-Consultancy/dlm-internal-utils.git#subdirectory=packages/csp
```

[Learn more about CSP](./packages/csp/README.md)

## 🔧 Repository Structure

```
dlm-internal-utils/
├── packages/
│   ├── csp/                # Connect SQL with Pandas package
│   │   ├── csp/            # Source code
│   │   ├── tests/          # Unit tests
│   │   ├── README.md       # Package documentation
│   │   └── setup.py        # Package installation
│   └── [future packages]   # More packages to be added
└── README.md               # This file
```

## 🚀 Installation

Each package can be installed individually using pip. See the respective package README for specific installation instructions.

## 🛠️ Development

To contribute to this repository:

1. Clone the repository
2. Create a new branch for your feature
3. Make your changes
4. Submit a pull request

## 📄 License

For internal use within DLM Consultancy only.