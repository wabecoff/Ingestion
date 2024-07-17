# tests/conftest.py

import sys
import os

# Insert the parent directory of the current file into sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
