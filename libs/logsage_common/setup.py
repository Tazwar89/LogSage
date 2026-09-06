from setuptools import setup, find_packages

setup(
    name="logsage_common",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "redis",
        "faiss-cpu",
        "sentence-transformers",
        "numpy",
    ],
)