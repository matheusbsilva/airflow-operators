from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import dlt
import duckdb
import pyarrow as pa
import yaml

from airflow.datasets import Dataset
from airflow.exceptions import AirflowException
from airflow.models.baseoperator import BaseOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.task_group import TaskGroup

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
    with duckdb.connect() as con:
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

    return {
        row[0]: {"data_type": "complex"}
        for row in rows
        if row[0] not in exclude_cols
    }


def _stream_table(
    source_conn_str: str,
    source_schema: str,
    table_name: str,
    exclude_cols: list[str],
    batch_size: int,
    incremental_key: str | None = None,
    start_value: str | None = None,
    end_value: str | None = None,
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

    conditions = []
    if incremental_key and start_value is not None:
        conditions.append(f"{incremental_key} >= '{start_value}'")
    if incremental_key and end_value is not None:
        conditions.append(f"{incremental_key} < '{end_value}'")
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    with duckdb.connect() as con:
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
        for batch in result.fetch_record_batch(rows_per_batch=batch_size):
            yield batch


def _make_resource(
    table_name: str,
    exclude_cols: list[str],
    incremental_key: str | None,
    primary_key: str,
    source_conn_str: str,
    source_schema: str,
    batch_size: int,
    jsonb_hints: dict[str, dict],
) -> dlt.sources.DltResource:
    """Dynamically build a ``dlt.resource`` for a single table.

    Two modes:
    * **Incremental** (``incremental_key`` present): uses dlt's delete+insert
      merge strategy.  ``allow_external_schedulers=True`` makes dlt defer to
      Airflow's ``data_interval_start`` / ``data_interval_end`` as the load
      window — Airflow owns scheduling, dlt does not persist its own cursor.
    * **Full refresh** (no ``incremental_key``): ``write_disposition="replace"``
      — the destination table is fully replaced on every run.
    """
    # Bind all table-specific values as default args to avoid late-binding.
    gen_defaults: dict = dict(
        _conn=source_conn_str,
        _schema=source_schema,
        _table=table_name,
        _excl=exclude_cols,
        _bs=batch_size,
        _ikey=incremental_key,
    )
    if incremental_key:
        gen_defaults["cursor"] = dlt.sources.incremental(
            incremental_key,
            initial_value="1970-01-01T00:00:00Z",
            # Defer the load window to Airflow's data_interval_start/end.
            # dlt reads these automatically from the Airflow task context.
            allow_external_schedulers=True,
        )

    def _gen(
        cursor=gen_defaults.get("cursor"),
        _conn=gen_defaults["_conn"],
        _schema=gen_defaults["_schema"],
        _table=gen_defaults["_table"],
        _excl=gen_defaults["_excl"],
        _bs=gen_defaults["_bs"],
        _ikey=gen_defaults["_ikey"],
    ):
        yield from _stream_table(
            _conn, _schema, _table, _excl, _bs,
            incremental_key=_ikey,
            start_value=cursor.start_value if cursor else None,
            end_value=cursor.end_value if cursor else None,
        )

    resource_kwargs: dict = dict(
        name=table_name,
        columns=jsonb_hints or None,
    )
    if incremental_key:
        resource_kwargs["primary_key"] = primary_key
        resource_kwargs["write_disposition"] = {"disposition": "merge", "strategy": "delete-insert"}
    else:
        resource_kwargs["write_disposition"] = "replace"

    return dlt.resource(_gen, **resource_kwargs)


# ---------------------------------------------------------------------------
# Airflow Operator  (single-table)
# ---------------------------------------------------------------------------

class PostgresCopyTable(BaseOperator):
    """Copy a single table from a source PostgreSQL database to a destination
    PostgreSQL database using `dlt <https://dlthub.com/>`_ for incremental
    state management and `DuckDB <https://duckdb.org/>`_ as the query engine.

    Intended to be instantiated once per table.  Use
    :func:`create_postgres_copy_task_group` to create a full TaskGroup of these
    operators from a YAML config file.

    **Incremental load and Airflow scheduling**

    When ``incremental_key`` is set, dlt uses ``allow_external_schedulers=True``
    so that Airflow's ``data_interval_start`` and ``data_interval_end`` control
    the load window — no separate dlt cursor state is persisted.  This integrates
    cleanly with Airflow's backfill and catchup mechanisms.

    **Full refresh**

    Pass ``full_refresh=True`` to replace the destination table entirely on the
    current run, regardless of ``incremental_key``.  Useful for one-off
    reloads or schema migrations.

    **Required packages**::

        dlt[postgres]   duckdb   pyarrow
        apache-airflow-providers-postgres

    :param source_conn_id: Airflow connection ID for the source PostgreSQL database.
    :param target_conn_id: Airflow connection ID for the destination PostgreSQL database.
    :param table_name: Name of the table to copy (same name used in both source and destination).
    :param dataset_name: Destination schema name (dlt ``dataset_name``).  Defaults to ``"public"``.
    :param source_schema: Source PostgreSQL schema.  Defaults to ``"public"``.
    :param incremental_key: Timestamp/datetime column used for incremental loading.
        Omit for full-replace behaviour.
    :param primary_key: Primary key column used for merge deduplication.  Defaults to ``"id"``.
    :param exclude_columns: List of column names to omit from the copy.
    :param batch_size: Rows per Arrow RecordBatch when streaming.  Defaults to ``50_000``.
    :param pipeline_name: dlt pipeline name for state storage.  Defaults to
        ``"pg_copy_{source_conn_id}_{table_name}"``.
    :param full_refresh: When ``True``, forces ``write_disposition="replace"`` for this
        run regardless of ``incremental_key``.  Defaults to ``False``.
    """

    template_fields = (
        "source_conn_id",
        "target_conn_id",
        "table_name",
        "dataset_name",
        "full_refresh",
    )

    def __init__(
        self,
        source_conn_id: str,
        target_conn_id: str,
        table_name: str,
        dataset_name: str = "public",
        source_schema: str = "public",
        incremental_key: str | None = None,
        primary_key: str = "id",
        exclude_columns: list[str] | None = None,
        batch_size: int = 50_000,
        pipeline_name: str | None = None,
        full_refresh: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.source_conn_id = source_conn_id
        self.target_conn_id = target_conn_id
        self.table_name = table_name
        self.dataset_name = dataset_name
        self.source_schema = source_schema
        self.incremental_key = incremental_key
        self.primary_key = primary_key
        self.exclude_columns = exclude_columns or []
        self.batch_size = batch_size
        self.pipeline_name = pipeline_name or f"pg_copy_{source_conn_id}_{table_name}"
        self.full_refresh = full_refresh

    # ------------------------------------------------------------------

    def execute(self, context) -> None:
        # 1. Resolve connection URIs from Airflow connections.
        source_hook = PostgresHook(postgres_conn_id=self.source_conn_id)
        target_hook = PostgresHook(postgres_conn_id=self.target_conn_id)
        source_conn_str: str = source_hook.get_uri()
        target_conn_str: str = target_hook.get_uri()

        # 2. Auto-detect JSONB columns in the source.
        jsonb_hints = _get_jsonb_columns(
            source_conn_str, self.source_schema, self.table_name, self.exclude_columns
        )
        if jsonb_hints:
            self.log.info(
                "Detected JSONB columns in '%s': %s", self.table_name, list(jsonb_hints)
            )

        # 3. Build the dlt resource for this table.
        self.log.info(
            "Preparing resource for table '%s' (incremental_key=%s, full_refresh=%s)",
            self.table_name,
            self.incremental_key or "none — full refresh",
            self.full_refresh,
        )

        resource = _make_resource(
            table_name=self.table_name,
            exclude_cols=self.exclude_columns,
            incremental_key=None if self.full_refresh else self.incremental_key,
            primary_key=self.primary_key,
            source_conn_str=source_conn_str,
            source_schema=self.source_schema,
            batch_size=self.batch_size,
            jsonb_hints=jsonb_hints,
        )

        # 4. Wrap resource in a dlt source.
        source_conn_id = self.source_conn_id

        @dlt.source(name=source_conn_id)
        def _source():
            return resource

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

        run_kwargs: dict = {"loader_file_format": "csv"}
        if self.full_refresh:
            run_kwargs["write_disposition"] = "replace"

        # loader_file_format="csv" uses PostgreSQL's COPY command for bulk inserts,
        # which is significantly faster than INSERT VALUES for large datasets.
        load_info = pipeline.run(_source(), **run_kwargs)

        self.log.info("Load complete:\n%s", load_info)

        # Surface any load errors as task failures.
        if load_info.has_failed_jobs:
            failed = [str(j) for p in load_info.load_packages for j in p.jobs.get("failed_jobs", [])]
            raise AirflowException(
                f"dlt pipeline '{self.pipeline_name}' had failed jobs:\n"
                + "\n".join(failed)
            )


# ---------------------------------------------------------------------------
# DAG-level factory  (creates a TaskGroup with one task per table)
# ---------------------------------------------------------------------------

def create_postgres_copy_task_group(
    group_id: str,
    source_conn_id: str,
    target_conn_id: str,
    config_path: str,
    dataset_name: str = "public",
    source_schema: str = "public",
    batch_size: int = 50_000,
    full_refresh: bool = False,
) -> TaskGroup:
    """Read a YAML config and build a :class:`TaskGroup` with one
    :class:`PostgresCopyTable` task per table entry.

    Each task is named ``copy_<table_name>`` inside the group.  On successful
    completion each task emits an Airflow :class:`~airflow.datasets.Dataset`
    event with URI ``postgres://<target_conn_id>/<dataset_name>/<table_name>``,
    enabling downstream DAGs to use data-aware scheduling.

    **YAML config format** (``config_path``)::

        <source_conn_id>:
          - table: <table_name>
            incremental_key: <timestamp_column>   # optional
            primary_key: <pk_column>              # optional, default 'id'
            exclude_columns:                      # optional
              - <column_name>

    **DAG usage example**::

        from plugins.operators.postgres_copy_table import create_postgres_copy_task_group

        with DAG("pg_copy", schedule="@daily", ...):
            create_postgres_copy_task_group(
                group_id="copy_tables",
                source_conn_id="source_postgres_conn",
                target_conn_id="dest_postgres_conn",
                config_path="/opt/airflow/include/config/tables.yml",
                dataset_name="analytics",
            )

    :param group_id: TaskGroup identifier shown in the Airflow UI.
    :param source_conn_id: Airflow connection ID for the source database.
        Must match a top-level key in the YAML config.
    :param target_conn_id: Airflow connection ID for the destination database.
    :param config_path: Path to the YAML config file.
    :param dataset_name: Destination schema name passed to each task.
    :param source_schema: Source schema name passed to each task.
    :param batch_size: Streaming batch size passed to each task.
    :param full_refresh: When ``True`` all tasks run as full refresh,
        ignoring ``incremental_key``.
    :returns: The constructed :class:`~airflow.utils.task_group.TaskGroup`.
    """
    config = yaml.safe_load(Path(config_path).read_text())
    tables: list[dict] = config.get(source_conn_id, [])
    if not tables:
        raise AirflowException(
            f"No tables configured for connection '{source_conn_id}' "
            f"in '{config_path}'."
        )

    with TaskGroup(group_id=group_id) as tg:
        for table_cfg in tables:
            table_name = table_cfg.get("table")
            if not table_name:
                raise AirflowException(
                    f"Each entry under '{source_conn_id}' must have a 'table' key."
                )

            PostgresCopyTable(
                task_id=f"copy_{table_name}",
                source_conn_id=source_conn_id,
                target_conn_id=target_conn_id,
                table_name=table_name,
                dataset_name=dataset_name,
                source_schema=source_schema,
                incremental_key=table_cfg.get("incremental_key"),
                primary_key=table_cfg.get("primary_key", "id"),
                exclude_columns=table_cfg.get("exclude_columns", []),
                batch_size=batch_size,
                full_refresh=full_refresh,
                outlets=[
                    Dataset(f"postgres://{target_conn_id}/{dataset_name}/{table_name}")
                ],
            )

    return tg
