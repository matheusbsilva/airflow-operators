"""Integration tests for PostgresCopyTable operator.

Each test class spins up two isolated PostgreSQL instances (via Docker /
testcontainers when available, local Postgres otherwise — see conftest.py)
and exercises the operator end-to-end against real databases.

The three behaviours under test
--------------------------------
1. **Correct schema loading** — column types (including JSONB) survive the
   round-trip; excluded columns are absent from the target.
2. **Incremental loading** — subsequent runs only process rows whose
   ``incremental_key`` value is newer than the last run's high-water mark;
   updated rows are replaced rather than duplicated.
3. **Full refresh on demand** — ``full_refresh=True`` replaces the entire
   target table, including rows that were deleted from the source.

Requirements (install once)
---------------------------
    pip install testcontainers[postgres] psycopg2-binary pytest dlt[postgres] duckdb pyarrow
    # For Docker path: docker pull postgres:15
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import psycopg2
import pytest
from psycopg2.extras import Json

from plugins.operators.postgres_copy_table import PostgresCopyTable


# ── Helpers ───────────────────────────────────────────────────────────────────


def _pg_uri(pg) -> str:
    """Return a plain ``postgresql://`` URI from any fixture that exposes
    ``get_connection_url()`` (works for both ``PostgresContainer`` and the
    local ``_LocalPg`` wrapper defined in conftest.py)."""
    return pg.get_connection_url().replace("+psycopg2", "")


def _connect(pg) -> psycopg2.extensions.connection:
    conn = psycopg2.connect(_pg_uri(pg))
    conn.autocommit = True
    return conn


def _uid() -> str:
    """Short, unique identifier used to name schemas so tests don't collide."""
    return uuid.uuid4().hex[:10]


def _run_operator(source_pg, target_pg, **op_kwargs) -> None:
    """Instantiate and execute a ``PostgresCopyTable`` operator, injecting the
    real Postgres URIs by patching ``PostgresHook.get_uri()``."""
    op_kwargs.setdefault("task_id", "test_copy")
    op_kwargs.setdefault("source_conn_id", "src")
    op_kwargs.setdefault("target_conn_id", "tgt")
    # Use a unique pipeline name by default so dlt state doesn't bleed between
    # tests.  Tests that need two runs to share state pass an explicit name.
    op_kwargs.setdefault("pipeline_name", f"test_{uuid.uuid4().hex[:12]}")

    op = PostgresCopyTable(**op_kwargs)

    src_uri = _pg_uri(source_pg)
    tgt_uri = _pg_uri(target_pg)

    with patch("plugins.operators.postgres_copy_table.PostgresHook") as mock_hook_cls:
        hook = MagicMock()
        hook.get_uri.side_effect = [src_uri, tgt_uri]
        mock_hook_cls.return_value = hook
        op.execute({})


# ── 1. Correct schema loading ─────────────────────────────────────────────────


class TestSchemaLoading:
    """Verify that tables are copied with the correct column types, including
    JSONB, and that excluded columns are absent from the target."""

    def test_basic_columns_are_copied_with_correct_values(
        self, source_pg, target_pg
    ):
        schema = f"s_{_uid()}"
        with _connect(source_pg) as conn, conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA {schema}")
            cur.execute(
                f"""
                CREATE TABLE {schema}.users (
                    id   SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    age  INT
                )
                """
            )
            cur.execute(
                f"INSERT INTO {schema}.users (name, age) VALUES (%s, %s), (%s, %s)",
                ("Alice", 30, "Bob", 25),
            )

        _run_operator(
            source_pg,
            target_pg,
            table_name="users",
            source_schema=schema,
            dataset_name=schema,
            full_refresh=True,
        )

        with _connect(target_pg) as conn, conn.cursor() as cur:
            cur.execute(f"SELECT name, age FROM {schema}.users ORDER BY name")
            rows = cur.fetchall()

        assert rows == [("Alice", 30), ("Bob", 25)]

    def test_jsonb_column_type_is_preserved_in_target(
        self, source_pg, target_pg
    ):
        schema = f"s_{_uid()}"
        with _connect(source_pg) as conn, conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA {schema}")
            cur.execute(
                f"""
                CREATE TABLE {schema}.events (
                    id      SERIAL PRIMARY KEY,
                    payload JSONB
                )
                """
            )
            cur.execute(
                f"INSERT INTO {schema}.events (payload) VALUES (%s), (%s)",
                (
                    Json({"type": "click", "x": 100}),
                    Json({"type": "view", "page": "/home"}),
                ),
            )

        _run_operator(
            source_pg,
            target_pg,
            table_name="events",
            source_schema=schema,
            dataset_name=schema,
            full_refresh=True,
        )

        with _connect(target_pg) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT udt_name
                FROM   information_schema.columns
                WHERE  table_schema = %s
                  AND  table_name   = 'events'
                  AND  column_name  = 'payload'
                """,
                (schema,),
            )
            row = cur.fetchone()

        assert row is not None, "payload column missing from target"
        assert row[0] == "jsonb", f"expected jsonb, got {row[0]}"

    def test_jsonb_data_values_round_trip_correctly(
        self, source_pg, target_pg
    ):
        schema = f"s_{_uid()}"
        payload = {"key": "value", "nested": {"n": 42}}

        with _connect(source_pg) as conn, conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA {schema}")
            cur.execute(f"CREATE TABLE {schema}.docs (id INT, data JSONB)")
            cur.execute(
                f"INSERT INTO {schema}.docs VALUES (%s, %s)",
                (1, Json(payload)),
            )

        _run_operator(
            source_pg,
            target_pg,
            table_name="docs",
            source_schema=schema,
            dataset_name=schema,
            full_refresh=True,
        )

        with _connect(target_pg) as conn, conn.cursor() as cur:
            cur.execute(f"SELECT data FROM {schema}.docs WHERE id = 1")
            row = cur.fetchone()

        assert row is not None
        assert row[0] == payload

    def test_excluded_columns_are_absent_from_target(
        self, source_pg, target_pg
    ):
        schema = f"s_{_uid()}"
        with _connect(source_pg) as conn, conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA {schema}")
            cur.execute(
                f"""
                CREATE TABLE {schema}.products (
                    id            SERIAL PRIMARY KEY,
                    name          TEXT,
                    internal_code TEXT
                )
                """
            )
            cur.execute(
                f"INSERT INTO {schema}.products (name, internal_code) VALUES (%s, %s)",
                ("Widget", "SECRET"),
            )

        _run_operator(
            source_pg,
            target_pg,
            table_name="products",
            source_schema=schema,
            dataset_name=schema,
            exclude_columns=["internal_code"],
            full_refresh=True,
        )

        with _connect(target_pg) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name
                FROM   information_schema.columns
                WHERE  table_schema = %s AND table_name = 'products'
                """,
                (schema,),
            )
            columns = {row[0] for row in cur.fetchall()}

        assert "internal_code" not in columns
        assert "name" in columns

    def test_multiple_jsonb_columns_all_have_correct_type(
        self, source_pg, target_pg
    ):
        schema = f"s_{_uid()}"
        with _connect(source_pg) as conn, conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA {schema}")
            cur.execute(
                f"""
                CREATE TABLE {schema}.mixed (
                    id       SERIAL PRIMARY KEY,
                    metadata JSONB,
                    settings JSONB,
                    label    TEXT
                )
                """
            )
            cur.execute(
                f"INSERT INTO {schema}.mixed (metadata, settings, label)"
                f" VALUES (%s, %s, %s)",
                (Json({"a": 1}), Json({"b": 2}), "hello"),
            )

        _run_operator(
            source_pg,
            target_pg,
            table_name="mixed",
            source_schema=schema,
            dataset_name=schema,
            full_refresh=True,
        )

        with _connect(target_pg) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name, udt_name
                FROM   information_schema.columns
                WHERE  table_schema = %s AND table_name = 'mixed'
                  AND  column_name IN ('metadata', 'settings')
                ORDER  BY column_name
                """,
                (schema,),
            )
            rows = cur.fetchall()

        assert len(rows) == 2
        assert all(udt == "jsonb" for _, udt in rows), (
            f"Expected both columns to be jsonb; got {rows}"
        )


# ── 2. Incremental loading ────────────────────────────────────────────────────


class TestIncrementalLoading:
    """Verify that the operator correctly identifies and loads only new or
    changed rows on subsequent runs, using dlt's cursor state."""

    def test_first_run_loads_all_existing_rows(
        self, source_pg, target_pg
    ):
        schema = f"s_{_uid()}"
        with _connect(source_pg) as conn, conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA {schema}")
            cur.execute(
                f"""
                CREATE TABLE {schema}.orders (
                    id         INT PRIMARY KEY,
                    item       TEXT,
                    updated_at TIMESTAMP WITH TIME ZONE
                )
                """
            )
            cur.execute(
                f"""
                INSERT INTO {schema}.orders VALUES
                    (1, 'Apple',  '2024-01-01 00:00:00+00'),
                    (2, 'Banana', '2024-01-02 00:00:00+00'),
                    (3, 'Cherry', '2024-01-03 00:00:00+00')
                """
            )

        _run_operator(
            source_pg,
            target_pg,
            table_name="orders",
            source_schema=schema,
            dataset_name=schema,
            incremental_key="updated_at",
            primary_key="id",
        )

        with _connect(target_pg) as conn, conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {schema}.orders")
            count = cur.fetchone()[0]

        assert count == 3

    def test_second_run_appends_only_new_rows(
        self, source_pg, target_pg
    ):
        schema = f"s_{_uid()}"
        pipeline_name = f"incr_{_uid()}"

        with _connect(source_pg) as conn, conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA {schema}")
            cur.execute(
                f"""
                CREATE TABLE {schema}.orders (
                    id         INT PRIMARY KEY,
                    item       TEXT,
                    updated_at TIMESTAMP WITH TIME ZONE
                )
                """
            )
            cur.execute(
                f"""
                INSERT INTO {schema}.orders VALUES
                    (1, 'Apple',  '2024-01-10 00:00:00+00'),
                    (2, 'Banana', '2024-01-11 00:00:00+00')
                """
            )

        common = dict(
            table_name="orders",
            source_schema=schema,
            dataset_name=schema,
            incremental_key="updated_at",
            primary_key="id",
            pipeline_name=pipeline_name,
        )

        # Run 1: bootstrap — load both existing rows.
        _run_operator(source_pg, target_pg, **common)

        # Add a row whose timestamp is strictly newer than the cursor.
        with _connect(source_pg) as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {schema}.orders VALUES
                    (3, 'Cherry', '2024-06-01 00:00:00+00')
                """
            )

        # Run 2: incremental — should pick up only the new row.
        _run_operator(source_pg, target_pg, **common)

        with _connect(target_pg) as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT id, item FROM {schema}.orders ORDER BY id"
            )
            rows = cur.fetchall()

        assert rows == [(1, "Apple"), (2, "Banana"), (3, "Cherry")]

    def test_old_rows_not_in_new_window_are_not_reloaded(
        self, source_pg, target_pg
    ):
        """Rows already past the cursor high-water mark must not cause the
        row count in the target to grow beyond what is in the source."""
        schema = f"s_{_uid()}"
        pipeline_name = f"incr_{_uid()}"

        with _connect(source_pg) as conn, conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA {schema}")
            cur.execute(
                f"""
                CREATE TABLE {schema}.logs (
                    id         INT PRIMARY KEY,
                    msg        TEXT,
                    updated_at TIMESTAMP WITH TIME ZONE
                )
                """
            )
            cur.execute(
                f"""
                INSERT INTO {schema}.logs VALUES
                    (1, 'first',  '2024-01-01 00:00:00+00'),
                    (2, 'second', '2024-01-02 00:00:00+00')
                """
            )

        common = dict(
            table_name="logs",
            source_schema=schema,
            dataset_name=schema,
            incremental_key="updated_at",
            primary_key="id",
            pipeline_name=pipeline_name,
        )

        _run_operator(source_pg, target_pg, **common)  # Run 1

        with _connect(source_pg) as conn, conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {schema}.logs VALUES (3, 'third', '2024-03-01 00:00:00+00')"
            )

        _run_operator(source_pg, target_pg, **common)  # Run 2

        with _connect(target_pg) as conn, conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {schema}.logs")
            count = cur.fetchone()[0]

        # Exactly 3 rows — no phantom duplicates from rows below the cursor.
        assert count == 3

    def test_updated_rows_are_merged_not_duplicated(
        self, source_pg, target_pg
    ):
        """When a source row is updated (newer ``updated_at``), the target must
        reflect the change without duplicating the row (delete-insert merge)."""
        schema = f"s_{_uid()}"
        pipeline_name = f"incr_{_uid()}"

        with _connect(source_pg) as conn, conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA {schema}")
            cur.execute(
                f"""
                CREATE TABLE {schema}.records (
                    id         INT PRIMARY KEY,
                    status     TEXT,
                    updated_at TIMESTAMP WITH TIME ZONE
                )
                """
            )
            cur.execute(
                f"""
                INSERT INTO {schema}.records VALUES
                    (1, 'pending', '2024-01-01 00:00:00+00'),
                    (2, 'pending', '2024-01-02 00:00:00+00')
                """
            )

        common = dict(
            table_name="records",
            source_schema=schema,
            dataset_name=schema,
            incremental_key="updated_at",
            primary_key="id",
            pipeline_name=pipeline_name,
        )

        _run_operator(source_pg, target_pg, **common)  # Run 1: load both rows

        # Update row 1 with a new timestamp so it falls in the next window.
        with _connect(source_pg) as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {schema}.records
                SET    status = 'done', updated_at = '2024-06-01 00:00:00+00'
                WHERE  id = 1
                """
            )

        _run_operator(source_pg, target_pg, **common)  # Run 2: picks up row 1

        with _connect(target_pg) as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT id, status FROM {schema}.records ORDER BY id"
            )
            rows = cur.fetchall()

        # Row 1 updated, row 2 unchanged, no duplicates.
        assert rows == [(1, "done"), (2, "pending")]


# ── 3. Full refresh on demand ─────────────────────────────────────────────────


class TestFullRefresh:
    """Verify that ``full_refresh=True`` causes the target table to be fully
    replaced by the current source contents."""

    def test_full_refresh_loads_all_source_rows(
        self, source_pg, target_pg
    ):
        schema = f"s_{_uid()}"
        with _connect(source_pg) as conn, conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA {schema}")
            cur.execute(f"CREATE TABLE {schema}.items (id INT, name TEXT)")
            cur.execute(
                f"INSERT INTO {schema}.items VALUES (1, 'A'), (2, 'B'), (3, 'C')"
            )

        _run_operator(
            source_pg,
            target_pg,
            table_name="items",
            source_schema=schema,
            dataset_name=schema,
            full_refresh=True,
        )

        with _connect(target_pg) as conn, conn.cursor() as cur:
            cur.execute(f"SELECT id, name FROM {schema}.items ORDER BY id")
            rows = cur.fetchall()

        assert rows == [(1, "A"), (2, "B"), (3, "C")]

    def test_full_refresh_removes_rows_deleted_from_source(
        self, source_pg, target_pg
    ):
        schema = f"s_{_uid()}"
        pipeline_name = f"full_{_uid()}"

        with _connect(source_pg) as conn, conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA {schema}")
            cur.execute(f"CREATE TABLE {schema}.items (id INT, name TEXT)")
            cur.execute(
                f"INSERT INTO {schema}.items VALUES (1, 'A'), (2, 'B'), (3, 'C')"
            )

        _run_operator(
            source_pg,
            target_pg,
            table_name="items",
            source_schema=schema,
            dataset_name=schema,
            pipeline_name=pipeline_name,
            full_refresh=True,
        )

        # Delete rows 1 and 2 from the source.
        with _connect(source_pg) as conn, conn.cursor() as cur:
            cur.execute(f"DELETE FROM {schema}.items WHERE id IN (1, 2)")

        # Full refresh must mirror the source exactly — target should have only row 3.
        _run_operator(
            source_pg,
            target_pg,
            table_name="items",
            source_schema=schema,
            dataset_name=schema,
            pipeline_name=pipeline_name,
            full_refresh=True,
        )

        with _connect(target_pg) as conn, conn.cursor() as cur:
            cur.execute(f"SELECT id, name FROM {schema}.items ORDER BY id")
            rows = cur.fetchall()

        assert rows == [(3, "C")]

    def test_full_refresh_overrides_incremental_key(
        self, source_pg, target_pg
    ):
        """``full_refresh=True`` must replace the table even when an
        ``incremental_key`` is configured, so that deleted rows are removed."""
        schema = f"s_{_uid()}"
        pipeline_name = f"full_{_uid()}"

        with _connect(source_pg) as conn, conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA {schema}")
            cur.execute(
                f"""
                CREATE TABLE {schema}.logs (
                    id         INT PRIMARY KEY,
                    msg        TEXT,
                    updated_at TIMESTAMP WITH TIME ZONE
                )
                """
            )
            cur.execute(
                f"""
                INSERT INTO {schema}.logs VALUES
                    (1, 'first',  '2024-01-01 00:00:00+00'),
                    (2, 'second', '2024-01-02 00:00:00+00'),
                    (3, 'third',  '2024-01-03 00:00:00+00')
                """
            )

        common = dict(
            table_name="logs",
            source_schema=schema,
            dataset_name=schema,
            incremental_key="updated_at",
            primary_key="id",
            pipeline_name=pipeline_name,
        )

        # First run: incremental load populates the target.
        _run_operator(source_pg, target_pg, **common)

        # Delete rows 1 and 2 from the source.
        with _connect(source_pg) as conn, conn.cursor() as cur:
            cur.execute(f"DELETE FROM {schema}.logs WHERE id IN (1, 2)")

        # Full refresh should see only row 3 despite incremental_key being set.
        _run_operator(source_pg, target_pg, **common, full_refresh=True)

        with _connect(target_pg) as conn, conn.cursor() as cur:
            cur.execute(f"SELECT id FROM {schema}.logs ORDER BY id")
            rows = cur.fetchall()

        assert rows == [(3,)]

    def test_full_refresh_second_run_contains_new_source_rows(
        self, source_pg, target_pg
    ):
        """Rows added to the source between runs must appear in the target
        after a full refresh — not just the rows from the first run."""
        schema = f"s_{_uid()}"
        pipeline_name = f"full_{_uid()}"

        with _connect(source_pg) as conn, conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA {schema}")
            cur.execute(f"CREATE TABLE {schema}.items (id INT, name TEXT)")
            cur.execute(f"INSERT INTO {schema}.items VALUES (1, 'A')")

        _run_operator(
            source_pg,
            target_pg,
            table_name="items",
            source_schema=schema,
            dataset_name=schema,
            pipeline_name=pipeline_name,
            full_refresh=True,
        )

        with _connect(source_pg) as conn, conn.cursor() as cur:
            cur.execute(f"INSERT INTO {schema}.items VALUES (2, 'B'), (3, 'C')")

        _run_operator(
            source_pg,
            target_pg,
            table_name="items",
            source_schema=schema,
            dataset_name=schema,
            pipeline_name=pipeline_name,
            full_refresh=True,
        )

        with _connect(target_pg) as conn, conn.cursor() as cur:
            cur.execute(f"SELECT id, name FROM {schema}.items ORDER BY id")
            rows = cur.fetchall()

        assert rows == [(1, "A"), (2, "B"), (3, "C")]
