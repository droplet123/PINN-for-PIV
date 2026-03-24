# conftest.py — project root
# Ensures that `src/` is importable as a package when running pytest
# from the project root directory (e.g. `pytest tests/`).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
