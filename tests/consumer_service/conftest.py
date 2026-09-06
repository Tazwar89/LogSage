import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "services" / "consumer_service"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "libs" / "logsage_common"))