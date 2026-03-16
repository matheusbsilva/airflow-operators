"""Tests for plugins/operators/postgres_copy_table.py

Covers three key behaviours:
1. Correct schema loading — JSONB columns are detected and mapped to the
   ``complex`` dlt type; excluded columns are omitted.
2. Incremental loading — the correct WHERE clause is built from the
   incremental cursor; the dlt resource uses the delete-insert merge strategy.
3. Full-refresh on demand — ``full_refresh=True`` forces
   ``write_disposition="replace"`` on the pipeline run regardless of whether
   an ``incremental_key`` is set.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from airflow.exceptions import AirflowException

from plugins.operators.postgres_copy_table import (
    PostgresCopyTable,
    _get_jsonb_columns,
    _make_resource,
    _stream_table,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _mock_duckdb_con(*, fetchall=None, batches=None):
    """Return a MagicMock that behaves like a ``duckdb.Connection`` used as a
    context manager (``with duckdb.connect() as con:``).

    ``__enter__`` returns the connection itself, matching DuckDB's real API.
    """
    con = MagicMock()
    con.__enter__ = MagicMock(return_value=con)
    con.__exit__ = MagicMock(return_value=False)
    if fetchall is not None:
        con.execute.return_value.fetchall.return_value = fetchall
    if batches is not None:
        con.execute.return_value.fetch_record_batch.return_value = iter(batches)
    return con


def _make_operator(**overrides):
    defaults = dict(
        task_id="test_task",
        source_conn_id="src_conn",
        target_conn_id="tgt_conn",
        table_name="my_table",
    )
    defaults.update(overrides)
    return PostgresCopyTable(**defaults)


def _setup_dlt_mocks(mock_dlt, mock_hook_cls, *, failed_jobs=False):
    """Wire up the dlt and PostgresHook mocks shared by execute() tests.

    Returns the ``mock_pipeline`` so callers can assert on ``run.call_args``.
    """
    mock_hook_cls.return_value.get_uri.side_effect = [
        "postgres://src",
        "postgres://tgt",
    ]

    mock_load_info = MagicMock()
    mock_load_info.has_failed_jobs = failed_jobs
    if failed_jobs:
        failed_job = MagicMock()
        mock_load_info.load_packages = [
            MagicMock(jobs={"failed_jobs": [failed_job]})
        ]
    else:
        mock_load_info.load_packages = []

    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = mock_load_info
    mock_dlt.pipeline.return_value = mock_pipeline
    # dlt.source(name=...) returns a decorator; make it a transparent pass-through
    mock_dlt.source.return_value = lambda fn: fn

    return mock_pipeline


# ── _get_jsonb_columns ────────────────────────────────────────────────────────


class TestGetJsonbColumns:
    """Unit tests for the JSONB-column discovery helper."""

    def test_returns_complex_hint_for_each_jsonb_column(self):
        con = _mock_duckdb_con(fetchall=[("meta",), ("cfg",)])
        with patch(
            "plugins.operators.postgres_copy_table.duckdb.connect", return_value=con
        ):
            result = _get_jsonb_columns("conn", "public", "tbl", [])

        assert result == {
            "meta": {"data_type": "complex"},
            "cfg": {"data_type": "complex"},
        }

    def test_excludes_columns_listed_in_exclude_cols(self):
        con = _mock_duckdb_con(fetchall=[("meta",), ("cfg",)])
        with patch(
            "plugins.operators.postgres_copy_table.duckdb.connect", return_value=con
        ):
            result = _get_jsonb_columns("conn", "public", "tbl", ["meta"])

        assert result == {"cfg": {"data_type": "complex"}}
        assert "meta" not in result

    def test_returns_empty_dict_when_no_jsonb_columns(self):
        con = _mock_duckdb_con(fetchall=[])
        with patch(
            "plugins.operators.postgres_copy_table.duckdb.connect", return_value=con
        ):
            result = _get_jsonb_columns("conn", "public", "tbl", [])

        assert result == {}

    def test_passes_correct_schema_and_table_as_bind_params(self):
        con = _mock_duckdb_con(fetchall=[])
        with patch(
            "plugins.operators.postgres_copy_table.duckdb.connect", return_value=con
        ):
            _get_jsonb_columns("conn", "my_schema", "my_table", [])

        # Second positional argument to con.execute() is the list of bind params
        bind_params = con.execute.call_args[0][1]
        assert bind_params == ["my_schema", "my_table"]


# ── _stream_table ─────────────────────────────────────────────────────────────


class TestStreamTable:
    """Unit tests for WHERE-clause generation and column exclusion in the
    streaming helper.  DuckDB I/O is mocked; we inspect the SQL handed to
    ``con.execute``."""

    def _sql(self, con):
        """Extract the SQL string from the most recent con.execute() call."""
        return con.execute.call_args[0][0]

    def test_full_table_produces_no_where_clause(self):
        con = _mock_duckdb_con(batches=[])
        with patch(
            "plugins.operators.postgres_copy_table.duckdb.connect", return_value=con
        ):
            list(_stream_table("conn", "public", "tbl", [], 1_000))

        assert "WHERE" not in self._sql(con)

    def test_incremental_where_clause_includes_start_and_end(self):
        con = _mock_duckdb_con(batches=[])
        with patch(
            "plugins.operators.postgres_copy_table.duckdb.connect", return_value=con
        ):
            list(
                _stream_table(
                    "conn",
                    "public",
                    "tbl",
                    [],
                    1_000,
                    incremental_key="updated_at",
                    start_value="2024-01-01T00:00:00Z",
                    end_value="2024-02-01T00:00:00Z",
                )
            )

        sql = self._sql(con)
        assert "WHERE" in sql
        assert "updated_at >= '2024-01-01T00:00:00Z'" in sql
        assert "updated_at < '2024-02-01T00:00:00Z'" in sql

    def test_incremental_with_only_start_value_omits_upper_bound(self):
        con = _mock_duckdb_con(batches=[])
        with patch(
            "plugins.operators.postgres_copy_table.duckdb.connect", return_value=con
        ):
            list(
                _stream_table(
                    "conn",
                    "public",
                    "tbl",
                    [],
                    1_000,
                    incremental_key="updated_at",
                    start_value="2024-01-01T00:00:00Z",
                    end_value=None,
                )
            )

        sql = self._sql(con)
        assert "updated_at >= '2024-01-01T00:00:00Z'" in sql
        assert " < " not in sql

    def test_excluded_columns_use_duckdb_exclude_syntax(self):
        con = _mock_duckdb_con(batches=[])
        with patch(
            "plugins.operators.postgres_copy_table.duckdb.connect", return_value=con
        ):
            list(_stream_table("conn", "public", "tbl", ["secret", "pii"], 1_000))

        assert "EXCLUDE (secret, pii)" in self._sql(con)

    def test_no_excluded_columns_selects_star(self):
        con = _mock_duckdb_con(batches=[])
        with patch(
            "plugins.operators.postgres_copy_table.duckdb.connect", return_value=con
        ):
            list(_stream_table("conn", "public", "tbl", [], 1_000))

        sql = self._sql(con)
        assert "SELECT *" in sql
        assert "EXCLUDE" not in sql

    def test_yields_all_record_batches_from_result(self):
        batch_a, batch_b = MagicMock(), MagicMock()
        con = _mock_duckdb_con(batches=[batch_a, batch_b])
        with patch(
            "plugins.operators.postgres_copy_table.duckdb.connect", return_value=con
        ):
            result = list(_stream_table("conn", "public", "tbl", [], 1_000))

        assert result == [batch_a, batch_b]


# ── _make_resource ────────────────────────────────────────────────────────────


class TestMakeResource:
    """Unit tests for the dlt resource factory.

    ``dlt`` is fully mocked so we only verify that the correct keyword
    arguments are forwarded to ``dlt.resource``.
    """

    @patch("plugins.operators.postgres_copy_table.dlt")
    def test_incremental_key_sets_delete_insert_merge_disposition(self, mock_dlt):
        mock_dlt.sources.incremental.return_value = MagicMock(
            start_value=None, end_value=None
        )

        _make_resource(
            table_name="orders",
            exclude_cols=[],
            incremental_key="updated_at",
            primary_key="id",
            source_conn_str="postgres://src",
            source_schema="public",
            batch_size=1_000,
            jsonb_hints={},
        )

        _, kwargs = mock_dlt.resource.call_args
        assert kwargs["write_disposition"] == {
            "disposition": "merge",
            "strategy": "delete-insert",
        }
        assert kwargs["primary_key"] == "id"

    @patch("plugins.operators.postgres_copy_table.dlt")
    def test_no_incremental_key_sets_replace_disposition(self, mock_dlt):
        _make_resource(
            table_name="orders",
            exclude_cols=[],
            incremental_key=None,
            primary_key="id",
            source_conn_str="postgres://src",
            source_schema="public",
            batch_size=1_000,
            jsonb_hints={},
        )

        _, kwargs = mock_dlt.resource.call_args
        assert kwargs["write_disposition"] == "replace"
        assert "primary_key" not in kwargs

    @patch("plugins.operators.postgres_copy_table.dlt")
    def test_jsonb_hints_are_passed_as_columns(self, mock_dlt):
        hints = {"payload": {"data_type": "complex"}}

        _make_resource(
            table_name="events",
            exclude_cols=[],
            incremental_key=None,
            primary_key="id",
            source_conn_str="postgres://src",
            source_schema="public",
            batch_size=1_000,
            jsonb_hints=hints,
        )

        _, kwargs = mock_dlt.resource.call_args
        assert kwargs["columns"] == hints

    @patch("plugins.operators.postgres_copy_table.dlt")
    def test_empty_jsonb_hints_passes_none_as_columns(self, mock_dlt):
        _make_resource(
            table_name="events",
            exclude_cols=[],
            incremental_key=None,
            primary_key="id",
            source_conn_str="postgres://src",
            source_schema="public",
            batch_size=1_000,
            jsonb_hints={},
        )

        _, kwargs = mock_dlt.resource.call_args
        assert kwargs["columns"] is None

    @patch("plugins.operators.postgres_copy_table.dlt")
    def test_incremental_resource_uses_allow_external_schedulers(self, mock_dlt):
        """Airflow's data_interval_start/end must control the load window."""
        mock_dlt.sources.incremental.return_value = MagicMock(
            start_value=None, end_value=None
        )

        _make_resource(
            table_name="orders",
            exclude_cols=[],
            incremental_key="updated_at",
            primary_key="id",
            source_conn_str="postgres://src",
            source_schema="public",
            batch_size=1_000,
            jsonb_hints={},
        )

        mock_dlt.sources.incremental.assert_called_once_with(
            "updated_at",
            initial_value="1970-01-01T00:00:00Z",
            allow_external_schedulers=True,
        )


# ── PostgresCopyTable.execute ─────────────────────────────────────────────────


class TestPostgresCopyTableExecute:
    """End-to-end tests for the operator's execute() method.

    All external I/O (Postgres connections, DuckDB, dlt pipeline) is mocked.
    """

    # ── 1. Correct schema loading ────────────────────────────────────────────

    @patch("plugins.operators.postgres_copy_table.PostgresHook")
    @patch("plugins.operators.postgres_copy_table._make_resource")
    @patch("plugins.operators.postgres_copy_table._get_jsonb_columns")
    @patch("plugins.operators.postgres_copy_table.dlt")
    def test_jsonb_hints_forwarded_to_make_resource(
        self, mock_dlt, mock_get_jsonb, mock_make_resource, mock_hook_cls
    ):
        hints = {"payload": {"data_type": "complex"}}
        mock_get_jsonb.return_value = hints
        _setup_dlt_mocks(mock_dlt, mock_hook_cls)

        _make_operator().execute({})

        mock_make_resource.assert_called_once()
        _, kwargs = mock_make_resource.call_args
        assert kwargs["jsonb_hints"] == hints

    @patch("plugins.operators.postgres_copy_table.PostgresHook")
    @patch("plugins.operators.postgres_copy_table._make_resource")
    @patch("plugins.operators.postgres_copy_table._get_jsonb_columns")
    @patch("plugins.operators.postgres_copy_table.dlt")
    def test_get_jsonb_columns_called_with_correct_args(
        self, mock_dlt, mock_get_jsonb, mock_make_resource, mock_hook_cls
    ):
        mock_get_jsonb.return_value = {}
        _setup_dlt_mocks(mock_dlt, mock_hook_cls)

        _make_operator(
            table_name="orders",
            source_schema="analytics",
            exclude_columns=["secret"],
        ).execute({})

        mock_get_jsonb.assert_called_once_with(
            "postgres://src", "analytics", "orders", ["secret"]
        )

    # ── 2. Incremental loading ───────────────────────────────────────────────

    @patch("plugins.operators.postgres_copy_table.PostgresHook")
    @patch("plugins.operators.postgres_copy_table._make_resource")
    @patch("plugins.operators.postgres_copy_table._get_jsonb_columns")
    @patch("plugins.operators.postgres_copy_table.dlt")
    def test_incremental_run_does_not_add_write_disposition_to_run_kwargs(
        self, mock_dlt, mock_get_jsonb, mock_make_resource, mock_hook_cls
    ):
        mock_get_jsonb.return_value = {}
        mock_pipeline = _setup_dlt_mocks(mock_dlt, mock_hook_cls)

        _make_operator(incremental_key="updated_at").execute({})

        run_kwargs = mock_pipeline.run.call_args[1]
        assert "write_disposition" not in run_kwargs
        assert run_kwargs["loader_file_format"] == "csv"

    @patch("plugins.operators.postgres_copy_table.PostgresHook")
    @patch("plugins.operators.postgres_copy_table._make_resource")
    @patch("plugins.operators.postgres_copy_table._get_jsonb_columns")
    @patch("plugins.operators.postgres_copy_table.dlt")
    def test_incremental_key_forwarded_to_make_resource(
        self, mock_dlt, mock_get_jsonb, mock_make_resource, mock_hook_cls
    ):
        mock_get_jsonb.return_value = {}
        _setup_dlt_mocks(mock_dlt, mock_hook_cls)

        _make_operator(incremental_key="updated_at", primary_key="order_id").execute({})

        _, kwargs = mock_make_resource.call_args
        assert kwargs["incremental_key"] == "updated_at"
        assert kwargs["primary_key"] == "order_id"

    # ── 3. Full refresh ──────────────────────────────────────────────────────

    @patch("plugins.operators.postgres_copy_table.PostgresHook")
    @patch("plugins.operators.postgres_copy_table._make_resource")
    @patch("plugins.operators.postgres_copy_table._get_jsonb_columns")
    @patch("plugins.operators.postgres_copy_table.dlt")
    def test_full_refresh_adds_replace_write_disposition(
        self, mock_dlt, mock_get_jsonb, mock_make_resource, mock_hook_cls
    ):
        mock_get_jsonb.return_value = {}
        mock_pipeline = _setup_dlt_mocks(mock_dlt, mock_hook_cls)

        _make_operator(full_refresh=True).execute({})

        run_kwargs = mock_pipeline.run.call_args[1]
        assert run_kwargs["write_disposition"] == "replace"

    @patch("plugins.operators.postgres_copy_table.PostgresHook")
    @patch("plugins.operators.postgres_copy_table._make_resource")
    @patch("plugins.operators.postgres_copy_table._get_jsonb_columns")
    @patch("plugins.operators.postgres_copy_table.dlt")
    def test_full_refresh_overrides_incremental_key(
        self, mock_dlt, mock_get_jsonb, mock_make_resource, mock_hook_cls
    ):
        """full_refresh=True must force replace even when incremental_key is set."""
        mock_get_jsonb.return_value = {}
        mock_pipeline = _setup_dlt_mocks(mock_dlt, mock_hook_cls)

        _make_operator(incremental_key="updated_at", full_refresh=True).execute({})

        run_kwargs = mock_pipeline.run.call_args[1]
        assert run_kwargs["write_disposition"] == "replace"

    @patch("plugins.operators.postgres_copy_table.PostgresHook")
    @patch("plugins.operators.postgres_copy_table._make_resource")
    @patch("plugins.operators.postgres_copy_table._get_jsonb_columns")
    @patch("plugins.operators.postgres_copy_table.dlt")
    def test_no_full_refresh_does_not_set_replace(
        self, mock_dlt, mock_get_jsonb, mock_make_resource, mock_hook_cls
    ):
        mock_get_jsonb.return_value = {}
        mock_pipeline = _setup_dlt_mocks(mock_dlt, mock_hook_cls)

        _make_operator(full_refresh=False).execute({})

        run_kwargs = mock_pipeline.run.call_args[1]
        assert "write_disposition" not in run_kwargs

    # ── Error handling ───────────────────────────────────────────────────────

    @patch("plugins.operators.postgres_copy_table.PostgresHook")
    @patch("plugins.operators.postgres_copy_table._make_resource")
    @patch("plugins.operators.postgres_copy_table._get_jsonb_columns")
    @patch("plugins.operators.postgres_copy_table.dlt")
    def test_failed_jobs_raise_airflow_exception(
        self, mock_dlt, mock_get_jsonb, mock_make_resource, mock_hook_cls
    ):
        mock_get_jsonb.return_value = {}
        _setup_dlt_mocks(mock_dlt, mock_hook_cls, failed_jobs=True)

        with pytest.raises(AirflowException, match="failed jobs"):
            _make_operator().execute({})

    @patch("plugins.operators.postgres_copy_table.PostgresHook")
    @patch("plugins.operators.postgres_copy_table._make_resource")
    @patch("plugins.operators.postgres_copy_table._get_jsonb_columns")
    @patch("plugins.operators.postgres_copy_table.dlt")
    def test_successful_run_does_not_raise(
        self, mock_dlt, mock_get_jsonb, mock_make_resource, mock_hook_cls
    ):
        mock_get_jsonb.return_value = {}
        _setup_dlt_mocks(mock_dlt, mock_hook_cls, failed_jobs=False)

        # Should complete without raising
        _make_operator().execute({})
