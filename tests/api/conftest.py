"""
Pytest configuration and shared fixtures.
"""
import pytest
import sys
from pathlib import Path

# Add project root to path (go up two levels: api -> tests -> root)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
