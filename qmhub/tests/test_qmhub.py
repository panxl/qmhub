"""
Unit and regression test for the qmhub package.
"""

# Import package, test suite, and other packages as needed
import qmhub
import pytest
import sys

def test_qmhub_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "qmhub" in sys.modules
