"""The cohort code reads ../config.yaml and ../weavehr-extensions/config.yml relative to the working directory, so
the tests run from Python/ (as the task scripts do)."""
import os
import sys
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parents[1]
os.chdir(PYTHON_DIR)
sys.path.insert(0, str(PYTHON_DIR))
