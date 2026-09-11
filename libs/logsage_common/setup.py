from setuptools import setup, find_packages

setup(
    name="logsage_common",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "redis",
    ],
    extras_require={
        # Only needed by services that actually build/query the FAISS index
        # (ingestion_service, analysis_service). consumer_service only uses
        # LogStore/redact and never imports vector_store, so it installs
        # logsage_common WITHOUT this extra -- skipping the ~700MB+
        # torch/sentence-transformers download chain entirely.
        "vector": [
            "sentence-transformers",
            "qdrant-client>=1.9.0",
            "pydantic>=2.5,<2.10",
            "numpy",
        ],
    },
)