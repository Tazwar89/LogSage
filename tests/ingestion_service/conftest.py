import sys
from pathlib import Path

# Add the repo's `services` directory (not services/ingestion_service itself)
# so tests import via the unique `ingestion_service.app...` namespace rather
# than a bare `app`, which would collide with analysis_service's own `app`
# package if both ever ended up on sys.path in the same run.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "services"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "libs" / "logsage_common"))