"""conftest.py – Shared pytest fixtures and configuration."""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so `src.*` imports resolve without
# installing the package in editable mode.
sys.path.insert(0, str(Path(__file__).parent))
