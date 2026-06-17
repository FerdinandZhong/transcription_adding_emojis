"""Make the ``src/`` layout importable in tests without installing the package."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
