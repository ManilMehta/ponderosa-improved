from setuptools import setup, find_packages

setup(
    name="ponderosa",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={"ponderosa": ["config/*.yaml", "config/*.def", "templates/*.html"]},
    install_requires=[
        "polars>=0.20.31",
        "numpy",
        "pandas",
        "scikit-learn",
        "networkx",
        "pyyaml",
        "matplotlib",
        "seaborn",
        "pyarrow",
        "scipy"
    ],
    python_requires=">=3.8",
)
