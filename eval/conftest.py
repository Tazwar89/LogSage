import os

# Must be set at import time: run_eval / agentic_pipeline / judge read MOCK_LLM
# when first imported, which happens during test collection, before any fixture.
# Assign (not setdefault) so a real MOCK_LLM=false in the shell/.env can't override it.
os.environ["MOCK_LLM"] = "true"