from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import dlt
import duckdb
import pyarrow as pa
import yaml

from airflow.exceptions import AirflowException
from airflow.models.baseoperator import BaseOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (module-level so they are independently testable)
# ---------------------------------------------------------------------------

def _get_jsonb_columns(
    source_conn_str: str,
    source_schema: str,
    table_name: str,
    exclude_cols: list[str],
) -> dict[str, dict]:
    """Return dlt column hints for every JSONB column in the source table.

    Queries ``information_schema.columns`` through DuckDB's postgres extension
    and returns a dict of the form ``{column_name: {"data_type": "complex"}}``
    that can be passed directly to ``dlt.resource(..., columns=...)``.

    The ``complex`` dlt type maps to ``JSONB`` when the destination is PostgreSQL.
    Columns present in *exclude_cols* are omitted.
    """
    con = duckdb.connect()
    try:
        con.sql("INSTALL postgres; LOAD postgres;")
        con.sql(
            f"ATTACH '{source_conn_str}' AS src (TYPE POSTGRES, READ_ONLY)"
        )
        rows = con.execute(
            """
            SELECT column_name
            FROM src.information_schema.columns
            WHERE table_schema = ?
              AND table_name   = ?
              AND udt_name     = 'jsonb'
            """,
            [source_schema, table_name],
        ).fetchall()
    finally:
        con.close()

    exclude_set = set(exclude_cols)
    return {
        row[0]: {"data_type": "complex"}
        for row in rows
        if row[0] not in exclude_set
    }


def _stream_table(
    source_conn_str: str,
    source_schema: str,
    table_name: str,
    exclude_cols: list[str],
    batch_size: int,
    incremental_key: str | None = None,
    start_value: str | None = None,
) -> Iterator[pa.RecordBatch]:
    """Open an in-memory DuckDB connection, attach the source PostgreSQL database,
    and stream the table in Arrow RecordBatch chunks.

    Uses ``EXCLUDE(...)`` for column exclusion so DuckDB resolves the column
    list server-side.  The optional *WHERE* clause pushes the incremental
    filter down to PostgreSQL via the postgres extension, minimising data
    transfer for large tables.

    Memory is bounded regardless of table size because data is never fully
    materialised — ``fetch_record_batch`` streams rows in chunks of
    *batch_size* rows.
    """
    col_selector = (
        f"* EXCLUDE ({', '.join(exclude_cols)})" if exclude_cols else "*"
    )
    where_clause = (
        f"WHERE {incremental_key} >= '{start_value}'"
        if incremental_key and start_value is not None
        else ""
    )

    con = duckdb.connect()
    try:
        con.sql("INSTALL postgres; LOAD postgres;")
        con.sql(
            f"ATTACH '{source_conn_str}' AS src"
            f" (TYPE POSTGRES, READ_ONLY, SCHEMA '{source_schema}')"
        )

        log.info(
            "Streaming %s.%s [%s] batch_size=%d",
            source_schema, table_name, where_clause or "full table", batch_size,
        )

        result = con.execute(
            f"SELECT {col_selector} FROM src.{table_name} {where_clause}"
        )
        reader = result.fetch_record_batch(rows_per_batch=batch_size)

        while True:
            try:
                yield reader.read_next_batch()
            except StopIteration:
                break
    finally:
        con.close()


def _make_resource(
    table_cfg: dict,
    source_conn_str: str,
    source_schema: str,
    batch_size: int,
    jsonb_hints: dict[str, dict],
) -> dlt.sources.DltResource:
    """Dynamically build a ``dlt.resource`` for a single table config entry.

    Two modes:
    * **Incremental** (``incremental_key`` present): uses dlt's delete+insert
      merge strategy.  The cursor value is persisted atomically with data in
      the destination DB so failed runs are safe to retry.
    * **Full refresh** (no ``incremental_key``): ``write_disposition="replace"``
      — the destination table is fully replaced on every run.

    Uses ``dlt.resource()`` as a function call (rather than a decorator) so
    that resources can be created inside a loop with per-table configuration.
    Each generator captures its arguments via Python default-parameter binding
    to avoid closure-over-loop-variable issues.
    """
    table_name = table_cfg["table"]
    exclude_cols = table_cfg.get("exclude_columns", [])
    incremental_key = table_cfg.get("incremental_key")
    primary_key = table_cfg.get("primary_key", "id")

    if incremental_key:
        # Bind all table-specific values as default args to avoid late-binding.
        def _gen(
            cursor=dlt.sources.incremental(
                incremental_key,
                initial_value="1970-01-01T00:00:00Z",
            ),
            _conn=source_conn_str,
            _schema=source_schema,
            _table=table_name,
            _excl=exclude_cols,
            _bs=batch_size,
            _ikey=incremental_key,
        ):
            yield from _stream_table(
                _conn, _schema, _table, _excl, _bs,
                incremental_key=_ikey,
                start_value=cursor.start_value,
            )

        return dlt.resource(
            _gen,
            name=table_name,
            primary_key=primary_key,
            write_disposition={"disposition": "merge", "strategy": "delete-insert"},
            columns=jsonb_hints or None,
        )
    else:
        # Bind all table-specific values as default args to avoid late-binding.
        def _gen(  # type: ignore[no-redef]
            _conn=source_conn_str,
            _schema=source_schema,
            _table=table_name,
            _excl=exclude_cols,
            _bs=batch_size,
        ):
            yield from _stream_table(_conn, _schema, _table, _excl, _bs)

        return dlt.resource(
            _gen,
            name=table_name,
            write_disposition="replace",
            columns=jsonb_hints or None,
        )


# ---------------------------------------------------------------------------
# Airflow Operator
# ---------------------------------------------------------------------------

class PostgresCopyTable(BaseOperator):
    """Copy tables from a source PostgreSQL database to a destination PostgreSQL
    database using `dlt <https://dlthub.com/>`_ for incremental state management
    and `DuckDB <https://duckdb.org/>`_ as the transformation and query engine.

    Tables to copy are defined in a YAML configuration file rather than as
    operator parameters — add new tables simply by editing the YAML, with no
    code changes required.

    **How it works**

    For each table listed under *source_conn_id* in the YAML:

    * DuckDB attaches the source PostgreSQL via the ``postgres`` extension and
      streams data as Arrow RecordBatches (memory-bounded, suitable for tables
      with hundreds of millions of rows).
    * JSONB columns are auto-detected via ``information_schema`` and preserved
      as ``JSONB`` in the destination.
    * If the table has an ``incremental_key``, dlt uses a delete+insert merge
      strategy: only rows where ``incremental_key >= last_cursor`` are fetched,
      and matching rows in the destination are deleted then re-inserted.  The
      cursor is stored atomically with the data in the destination — if a run
      fails, the next run replays from the last successful cursor.
    * Tables without an ``incremental_key`` are fully replaced on every run.

    **YAML config format** (``config_path``)::

        <airflow_connection_name>:
          - table: <table_name>
            incremental_key: <timestamp_column>   # optional
            primary_key: <pk_column>              # optional, default 'id'
            exclude_columns:                      # optional
              - <column_name>

    **Required packages**::

        dlt[postgres]   duckdb   pyarrow   pyyaml
        apache-airflow-providers-postgres

    :param source_conn_id: Airflow connection ID for the source PostgreSQL database.
        Must match a top-level key in the YAML config file.
    :param target_conn_id: Airflow connection ID for the destination PostgreSQL database.
    :param config_path: Absolute (or Airflow-relative) path to the YAML config file.
    :param dataset_name: Destination schema name (dlt ``dataset_name``).  Defaults
        to ``"public"``.
    :param source_schema: Source PostgreSQL schema to read tables from.  Defaults
        to ``"public"``.
    :param batch_size: Number of rows per Arrow RecordBatch when streaming large
        tables.  Defaults to ``50_000``.
    :param pipeline_name: dlt pipeline name used for state storage.  Defaults to
        ``"pg_copy_<source_conn_id>"``.  Override when running multiple pipelines
        writing to the same destination schema to avoid state conflicts.
    """

    template_fields = ("source_conn_id", "target_conn_id", "config_path", "dataset_name")

    def __init__(
        self,
        source_conn_id: str,
        target_conn_id: str,
        config_path: str,
        dataset_name: str = "public",
        source_schema: str = "public",
        batch_size: int = 50_000,
        pipeline_name: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.source_conn_id = source_conn_id
        self.target_conn_id = target_conn_id
        self.config_path = config_path
        self.dataset_name = dataset_name
        self.source_schema = source_schema
        self.batch_size = batch_size
        self.pipeline_name = pipeline_name or f"pg_copy_{source_conn_id}"

    # ------------------------------------------------------------------

    def execute(self, context) -> None:
        # 1. Load YAML config and extract the table list for this source.
        config = yaml.safe_load(Path(self.config_path).read_text())
        tables: list[dict] = config.get(self.source_conn_id, [])
        if not tables:
            raise AirflowException(
                f"No tables configured for connection '{self.source_conn_id}' "
                f"in '{self.config_path}'."
            )

        # 2. Resolve connection URIs from Airflow connections.
        source_hook = PostgresHook(postgres_conn_id=self.source_conn_id)
        target_hook = PostgresHook(postgres_conn_id=self.target_conn_id)
        source_conn_str: str = source_hook.get_uri()
        target_conn_str: str = target_hook.get_uri()

        # 3. Build dlt resources — one per table.
        resources = []
        for table_cfg in tables:
            table_name = table_cfg.get("table")
            if not table_name:
                raise AirflowException(
                    f"Each entry in '{self.source_conn_id}' must have a 'table' key."
                )

            exclude_cols = table_cfg.get("exclude_columns", [])

            self.log.info(
                "Preparing resource for table '%s' (incremental_key=%s)",
                table_name,
                table_cfg.get("incremental_key", "none — full refresh"),
            )

            jsonb_hints = _get_jsonb_columns(
                source_conn_str, self.source_schema, table_name, exclude_cols
            )
            if jsonb_hints:
                self.log.info(
                    "Detected JSONB columns in '%s': %s", table_name, list(jsonb_hints)
                )

            resource = _make_resource(
                table_cfg, source_conn_str, self.source_schema,
                self.batch_size, jsonb_hints,
            )
            resources.append(resource)

        # 4. Wrap resources in a dlt source.
        @dlt.source(name=self.source_conn_id)
        def _source():
            return resources

        # 5. Configure and run the dlt pipeline.
        pipeline = dlt.pipeline(
            pipeline_name=self.pipeline_name,
            destination=dlt.destinations.postgres(credentials=target_conn_str),
            dataset_name=self.dataset_name,
        )

        self.log.info(
            "Running dlt pipeline '%s' → schema '%s' on '%s'",
            self.pipeline_name, self.dataset_name, self.target_conn_id,
        )

        # loader_file_format="csv" uses PostgreSQL's COPY command for bulk inserts,
        # which is significantly faster than INSERT VALUES for large datasets.
        load_info = pipeline.run(_source(), loader_file_format="csv")

        self.log.info("Load complete:\n%s", load_info)

        # Surface any load errors as task failures.
        if load_info.has_failed_jobs:
            failed = [str(j) for p in load_info.load_packages for j in p.jobs.get("failed_jobs", [])]
            raise AirflowException(
                f"dlt pipeline '{self.pipeline_name}' had failed jobs:\n"
                + "\n".join(failed)
            )
