"""
Shared pytest configuration.

Point the API's SQLite persistence at a throwaway temp database BEFORE any
test imports the app, so tests never touch the real data/spike_sense.db.
The env var is read when src.api.database first creates its engine.
"""

import os
import tempfile
from pathlib import Path

# Use a dedicated temp file DB for the whole test session.
_TEST_DB = Path(tempfile.gettempdir()) / "spike_sense_test.db"
os.environ["SPIKE_SENSE_DB_URL"] = f"sqlite:///{_TEST_DB}"

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_db():
    """Reset the database tables before each test for isolation."""
    from src.api import database as db

    db.init_db()
    db.reset_db()
    yield
