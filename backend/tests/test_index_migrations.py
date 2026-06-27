"""Tests for the index-adding migration introduced for PR #666.

Verifies that ``_migrate_add_indexes``:

1. Creates the expected index on every table/column it knows about.
2. Is idempotent — calling it twice does not error or duplicate work.
3. Skips silently when a table doesn't yet exist (e.g. partial DB).
"""

from sqlalchemy import create_engine, text

from backend.database.migrations import _migrate_add_indexes

EXPECTED_INDEXES = [
    ("ix_generations_profile_id", "generations", "profile_id"),
    ("ix_generations_created_at", "generations", "created_at"),
    ("ix_generations_status", "generations", "status"),
    ("ix_generations_is_favorited", "generations", "is_favorited"),
    ("ix_story_items_story_id", "story_items", "story_id"),
    ("ix_story_items_generation_id", "story_items", "generation_id"),
    ("ix_generation_versions_generation_id", "generation_versions", "generation_id"),
    ("ix_profile_samples_profile_id", "profile_samples", "profile_id"),
    ("ix_profile_samples_sort_order", "profile_samples", "sort_order"),
    ("ix_captures_created_at", "captures", "created_at"),
    ("ix_channel_device_mappings_channel_id", "channel_device_mappings", "channel_id"),
]


def _create_minimal_schema(engine):
    """Create just the tables needed to exercise the migration."""
    with engine.connect() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE generations (
                    id TEXT PRIMARY KEY,
                    profile_id TEXT NOT NULL,
                    created_at TIMESTAMP,
                    status TEXT DEFAULT 'completed',
                    is_favorited BOOLEAN DEFAULT 0
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE story_items (
                    id TEXT PRIMARY KEY,
                    story_id TEXT NOT NULL,
                    generation_id TEXT NOT NULL
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE generation_versions (
                    id TEXT PRIMARY KEY,
                    generation_id TEXT NOT NULL
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE profile_samples (
                    id TEXT PRIMARY KEY,
                    profile_id TEXT NOT NULL,
                    sort_order INTEGER NOT NULL DEFAULT 0
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE captures (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMP
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE channel_device_mappings (
                    id TEXT PRIMARY KEY,
                    channel_id TEXT NOT NULL
                )
                """
            )
        )
        conn.commit()


def _list_indexes(engine, table: str) -> set[str]:
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=:t"),
            {"t": table},
        ).fetchall()
    return {r[0] for r in rows}


def test_migrate_add_indexes_creates_all_expected_indexes():
    """Every documented index is created on its table."""
    engine = create_engine("sqlite:///:memory:")
    _create_minimal_schema(engine)
    tables = {
        "generations",
        "story_items",
        "generation_versions",
        "profile_samples",
        "captures",
        "channel_device_mappings",
    }

    _migrate_add_indexes(engine, tables)

    for index_name, table, _column in EXPECTED_INDEXES:
        assert index_name in _list_indexes(engine, table), (
            f"expected index {index_name} on {table} to be created"
        )


def test_migrate_add_indexes_is_idempotent():
    """Calling the migration twice must not error — uses CREATE INDEX IF NOT EXISTS."""
    engine = create_engine("sqlite:///:memory:")
    _create_minimal_schema(engine)
    tables = {
        "generations",
        "story_items",
        "generation_versions",
        "profile_samples",
        "captures",
        "channel_device_mappings",
    }

    _migrate_add_indexes(engine, tables)
    # Second call: must succeed (CREATE INDEX IF NOT EXISTS), and must not
    # create duplicate index entries.
    _migrate_add_indexes(engine, tables)

    for index_name, table, _column in EXPECTED_INDEXES:
        indexes = _list_indexes(engine, table)
        matching = [name for name in indexes if name == index_name]
        assert len(matching) == 1, (
            f"index {index_name} should appear exactly once, got {indexes}"
        )


def test_migrate_add_indexes_skips_missing_tables_silently():
    """If a table doesn't exist yet, the migration should skip without raising."""
    engine = create_engine("sqlite:///:memory:")
    # Only create one table; the others are missing.
    with engine.connect() as conn:
        conn.execute(
            text("CREATE TABLE generations (id TEXT PRIMARY KEY, profile_id TEXT, "
                 "created_at TIMESTAMP, status TEXT, is_favorited BOOLEAN DEFAULT 0)")
        )
        conn.commit()

    # Report only the one table we created — migration must not assume others.
    tables = {"generations"}
    _migrate_add_indexes(engine, tables)

    # Index on the present table should be created.
    assert "ix_generations_profile_id" in _list_indexes(engine, "generations")


def test_migrate_add_indexes_no_op_when_table_set_empty():
    """If the engine has no tables at all, the migration should be a safe no-op."""
    engine = create_engine("sqlite:///:memory:")
    _migrate_add_indexes(engine, set())  # must not raise
