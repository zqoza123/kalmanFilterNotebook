from setuptools import find_packages, setup

setup(
    name="kalman-pairs-trading",
    version="0.1.0",
    description="Notebook-first Kalman filter pairs trading project",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "matplotlib>=3.7",
        "yfinance>=0.2.40",
        "statsmodels>=0.14",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0",
            "pytest-cov>=5.0",
        ]
    },
)
