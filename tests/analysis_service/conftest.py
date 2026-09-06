import sys
from pathlib import Path

# See tests/ingestion-service/conftest.py for why this must never run in
# the same pytest process as ingestion-service's or consumer-service's tests.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "services" / "analysis-service"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "libs" / "logsage_common"))