from __future__ import annotations

# Disable dlt plugin auto-discovery before any other import.
# Some environments have distributions with broken metadata (e.g. PyJWT
# installed via Debian packages), which trips dlt's plugin loader with an
# AttributeError.  Disabling plugin discovery is safe for testing.
import os as _os
_os.environ.setdefault("DLT_DISABLE_PLUGINS", "true")

import json
import re
import uuid
from typing import Any, Generator, Iterator

import psycopg2
import pyarrow as pa
import pytest


# ── Postgres service abstraction ──────────────────────────────────────────────


class _LocalPg:
    """Stand-in for ``testcontainers.postgres.PostgresContainer``.

    Wraps an already-running Postgres instance with the same
    ``get_connection_url()`` interface so test code is identical in both
    execution modes.
    """

    def __init__(self, host: str, port: int, user: str, password: str, db: str):
        self._url = f"postgresql://{user}:{password}@{host}:{port}/{db}"

    def get_connection_url(self) -> str:
        return self._url


def _try_docker_postgres() -> object | None:
    """Return a started ``PostgresContainer`` or ``None`` if unavailable."""
    try:
        import docker
        from testcontainers.postgres import PostgresContainer

        docker.from_env().images.get("postgres:15")  # raises if image absent
        container = PostgresContainer("postgres:15")
        container.start()
        return container
    except Exception:
        return None


_LOCAL_SRC = _LocalPg("localhost", 5432, "testuser", "testpass", "testdb_src")
_LOCAL_TGT = _LocalPg("localhost", 5432, "testuser", "testpass", "testdb_tgt")


@pytest.fixture(scope="session")
def source_pg():
    container = _try_docker_postgres()
    if container:
        yield container
        container.stop()
    else:
        yield _LOCAL_SRC


@pytest.fixture(scope="session")
def target_pg():
    container = _try_docker_postgres()
    if container:
        yield container
        container.stop()
    else:
        yield _LOCAL_TGT


# ── DuckDB postgres extension / psycopg2 mock ─────────────────────────────────


def _try_install_duckdb_postgres() -> bool:
    """Return True if DuckDB's postgres extension installs successfully."""
    try:
        import duckdb

        with duckdb.connect() as con:
            con.sql("INSTALL postgres; LOAD postgres;")
        return True
    except Exception:
        return False


class _MockResult:
    """Wraps a psycopg2 cursor with the DuckDB result API used by the operator."""

    def __init__(self, cursor: Any) -> None:
        self._cur = cursor

    def fetchall(self) -> list:
        return self._cur.fetchall()

    def fetch_record_batch(
        self, rows_per_batch: int = 10_000
    ) -> Iterator[pa.RecordBatch]:
        columns = [desc[0] for desc in self._cur.description]
        while True:
            rows = self._cur.fetchmany(rows_per_batch)
            if not rows:
                break
            arrays: list[pa.Array] = []
            for col_idx, col_name in enumerate(columns):
                values = [row[col_idx] for row in rows]
                # JSONB comes back from psycopg2 as Python dicts/lists;
                # serialise to JSON strings so pyarrow uses the utf8 type
                # (matching what DuckDB returns for JSONB columns).
                if any(isinstance(v, (dict, list)) for v in values if v is not None):
                    values = [
                        json.dumps(v) if v is not None else None for v in values
                    ]
                arrays.append(pa.array(values))
            yield pa.RecordBatch.from_arrays(arrays, names=columns)


class _MockCon:
    """psycopg2-backed drop-in for a DuckDB connection.

    Supports the exact DuckDB API surface used by the operator:
    * ``con.sql("INSTALL postgres; LOAD postgres;")``    → no-op
    * ``con.sql("ATTACH '...' AS src (TYPE POSTGRES, ...)")``    → open psycopg2
    * ``con.execute("SELECT … FROM src.tbl WHERE …", params)``  → rewrite & run
    """

    def __init__(self) -> None:
        self._conn: psycopg2.extensions.connection | None = None
        self._schema: str = "public"

    # ── context manager ──────────────────────────────────────────────────────

    def __enter__(self) -> "_MockCon":
        return self

    def __exit__(self, *_: Any) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── DuckDB-compatible API ─────────────────────────────────────────────────

    def sql(self, query: str) -> "_MockCon":
        """Handle DDL-like statements; parse ATTACH to open a psycopg2 conn."""
        attach = re.search(r"ATTACH\s+'([^']+)'\s+AS\s+src", query, re.IGNORECASE)
        if attach:
            uri = attach.group(1)
            schema_m = re.search(r"SCHEMA\s+'([^']+)'", query, re.IGNORECASE)
            self._schema = schema_m.group(1) if schema_m else "public"
            self._conn = psycopg2.connect(uri)
        # INSTALL / LOAD / other DDL: silently ignored
        return self

    def execute(self, query: str, params: list | None = None) -> _MockResult:
        assert self._conn is not None, "execute() called before ATTACH"
        sql = self._rewrite(query)
        cur = self._conn.cursor()
        cur.execute(sql, params or [])
        return _MockResult(cur)

    # ── SQL rewriter ──────────────────────────────────────────────────────────

    def _rewrite(self, query: str) -> str:
        """Translate DuckDB SQL dialects to standard PostgreSQL."""
        q = query

        # 1. "src.information_schema" → "information_schema"
        #    (DuckDB prefix for the attached DB's system catalog)
        q = re.sub(r"\bsrc\.information_schema\b", "information_schema", q, flags=re.I)

        # 2. "src.<table>" → "<schema>.<table>"
        #    (DuckDB prefix for tables in the attached DB)
        q = re.sub(r"\bsrc\.(\w+)\b", rf"{self._schema}.\1", q)

        # 3. "* EXCLUDE (col1, col2)" → explicit column list
        #    (DuckDB syntax not supported by PostgreSQL)
        exclude_m = re.search(r"\*\s+EXCLUDE\s*\(([^)]+)\)", q, re.I)
        if exclude_m and self._conn:
            excluded = {c.strip() for c in exclude_m.group(1).split(",")}
            # Extract table name from the FROM clause
            from_m = re.search(
                rf"\bFROM\s+{re.escape(self._schema)}\.(\w+)\b", q, re.I
            )
            if from_m:
                table = from_m.group(1)
                cur = self._conn.cursor()
                cur.execute(
                    "SELECT column_name FROM information_schema.columns"
                    " WHERE table_schema = %s AND table_name = %s"
                    " ORDER BY ordinal_position",
                    (self._schema, table),
                )
                cols = [r[0] for r in cur.fetchall() if r[0] not in excluded]
                q = q.replace(exclude_m.group(0), ", ".join(cols))

        # 4. DuckDB "?" placeholders → psycopg2 "%s"
        q = q.replace("?", "%s")

        return q


def _patch_duckdb() -> None:
    """Replace ``duckdb.connect`` with a factory that returns ``_MockCon``."""
    import duckdb as _duckdb

    _duckdb.connect = lambda *a, **kw: _MockCon()  # type: ignore[method-assign]


# ── Session-level autouse fixture ─────────────────────────────────────────────


@pytest.fixture(scope="session", autouse=True)
def duckdb_postgres_setup() -> Generator[None, None, None]:
    """Ensure the DuckDB postgres extension (or its mock) is ready.

    Tries the real extension first.  Falls back transparently to a
    psycopg2-backed mock that reproduces the exact API patterns the operator
    uses, so the integration tests run end-to-end against real Postgres
    databases regardless of network access to the DuckDB extension CDN.
    """
    if not _try_install_duckdb_postgres():
        _patch_duckdb()
    yield
