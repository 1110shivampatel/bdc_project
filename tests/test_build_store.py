import duckdb

from src.config import get_path
from src.data.build_store import build_numeric_store


def test_build_numeric_store_for_single_period():
    """
    Run the numeric store builder for a single known period and ensure
    the DuckDB table is created with at least one row.

    This uses the real raw data but only touches the configured DuckDB file.
    """
    # Limit to a single, small-ish period to keep the test fast.
    build_numeric_store(periods=["2025_Q1"])

    duckdb_path = get_path("duckdb_path")
    assert duckdb_path.exists()

    conn = duckdb.connect(str(duckdb_path))
    try:
        count = conn.execute("SELECT COUNT(*) FROM numeric_facts WHERE period = '2025_Q1'").fetchone()[0]
        assert count > 0
    finally:
        conn.close()


