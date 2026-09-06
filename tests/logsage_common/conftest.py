import sys
from pathlib import Path

# Makes `logsage_common` importable without a full pip install -e,
# by pointing directly at the package source.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "libs" / "logsage_common"))