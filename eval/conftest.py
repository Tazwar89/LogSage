"""
Forces the offline path for every test in eval/ *before* any test module is
imported. run_eval, agentic_pipeline and judge all read MOCK_LLM at import time,
so setting it inside a test module after its imports is too late: the real
QdrantVectorStore/SentenceTransformer path loads, and on the CI runner
importing torch/triton through sentence_transformers segfaults.
"""
import os

os.environ["MOCK_LLM"] = "true"