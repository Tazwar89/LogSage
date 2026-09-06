import sys
from pathlib import Path

# Only this service's app/ directory goes on sys.path -- never combine with
# analysis_service's conftest.py in the same pytest run, since both define
# a top-level package literally named `app` and will silently shadow each
# other (confirmed: whichever is added to sys.path first wins).
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "services" / "ingestion_service"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "libs" / "logsage_common"))