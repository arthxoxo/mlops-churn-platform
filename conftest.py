"""
Root conftest.py — ensures pytest can discover the src package.
"""

import sys
from pathlib import Path

# Add project root to sys.path so 'from src.<module>' imports work
sys.path.insert(0, str(Path(__file__).parent))
