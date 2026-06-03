"""Shared pytest config. Ensures the repo root is importable so `import app.*`
works regardless of the working directory pytest is invoked from."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
