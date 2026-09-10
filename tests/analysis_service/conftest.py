import sys
from pathlib import Path

# See tests/ingestion_service/conftest.py for why this must never run in
# the same pytest process as ingestion_service's or consumer_service's tests.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "services"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "libs" / "logsage_common"))