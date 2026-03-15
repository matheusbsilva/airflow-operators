from typing import Callable, List, Literal, Optional
import os
from pathlib import Path

import duckdb

from airflow.exceptions import AirflowException
from airflow.models.baseoperator import BaseOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol


class WriteStrategy(Protocol):
    def write(
        self,
        conn: duckdb.DuckDBPyConnection,
        target_table_path: str,
        storage_filepath: str,
    ) -> None: ...


class ReplaceStrategy:
    def __init__(self, log) -> None:
        self._log = log

    def write(
        self,
        conn: duckdb.DuckDBPyConnection,
        target_table_path: str,
        storage_filepath: str,
    ) -> None:
        conn.execute(f"""
            CREATE OR REPLACE TABLE {target_table_path} AS
            SELECT * FROM '{storage_filepath}'
            WITH NO DATA
        """)
        self._log.info(f"Inserting values from {storage_filepath} into {target_table_path}")
        conn.execute(f"""
            INSERT INTO {target_table_path} BY NAME (
                SELECT * FROM '{storage_filepath}'
            )
        """)


class TruncateStrategy:
    def __init__(self, log, sync_fn: Callable) -> None:
        self._log = log
        self._sync_fn = sync_fn

    def write(
        self,
        conn: duckdb.DuckDBPyConnection,
        target_table_path: str,
        storage_filepath: str,
    ) -> None:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {target_table_path} AS
            SELECT * FROM '{storage_filepath}'
            WITH NO DATA
        """)
        self._sync_fn(conn, target_table_path, storage_filepath)
        self._log.info(f"Truncating table {target_table_path}")
        conn.execute(f'TRUNCATE {target_table_path}')
        self._log.info(f"Inserting values from {storage_filepath} into {target_table_path}")
        conn.execute(f"""
            INSERT INTO {target_table_path} BY NAME (
                SELECT * FROM '{storage_filepath}'
            )
        """)


class TempStorage(Protocol):
    @property
    def filepath(self) -> str: ...
    def setup(self, conn: duckdb.DuckDBPyConnection) -> None: ...
    def copy(self, conn: duckdb.DuckDBPyConnection, source_query: str) -> None: ...
    def cleanup(self) -> None: ...


class LocalTempStorage:
    def __init__(self, local_storage_path: str, source_table: str, log) -> None:
        self._log = log
        Path(local_storage_path).mkdir(parents=True, exist_ok=True)
        self._filepath = os.path.join(local_storage_path, f'{source_table}.parquet')

    @property
    def filepath(self) -> str:
        return self._filepath

    def setup(self, conn: duckdb.DuckDBPyConnection) -> None:
        pass

    def copy(self, conn: duckdb.DuckDBPyConnection, source_query: str) -> None:
        self._log.info(f"Copying data to local storage: {self._filepath}")
        conn.execute(f"""
            COPY ({source_query}) TO '{self._filepath}' (
                FORMAT parquet,
                COMPRESSION zstd
            )
        """)

    def cleanup(self) -> None:
        try:
            if os.path.exists(self._filepath):
                os.remove(self._filepath)
                self._log.info(f"Cleaned up local file: {self._filepath}")
        except OSError as e:
            self._log.warning(f"Failed to cleanup local file {self._filepath}: {e}")


class S3TempStorage:
    def __init__(self, bucket_uri: str, source_table: str, aws_conn_id: str, log) -> None:
        self._log = log
        self._aws_conn_id = aws_conn_id
        self._filepath = f'{bucket_uri}/{source_table}.parquet'

    @property
    def filepath(self) -> str:
        return self._filepath

    def setup(self, conn: duckdb.DuckDBPyConnection) -> None:
        conn.execute('INSTALL httpfs; LOAD httpfs;')

        s3_hook = S3Hook(aws_conn_id=self._aws_conn_id)
        region_name = s3_hook.region_name
        credentials = s3_hook.get_credentials()

        conn.execute(f"""
            CREATE SECRET (
                TYPE s3,
                KEY_ID '{credentials.access_key}',
                SECRET '{credentials.secret_key}',
                REGION '{region_name}'
            );
        """)

    def copy(self, conn: duckdb.DuckDBPyConnection, source_query: str) -> None:
        self._log.info(f"Copying data to S3: {self._filepath}")
        conn.execute(f"""
            COPY ({source_query}) TO '{self._filepath}' (
                FORMAT parquet,
                COMPRESSION zstd
            )
        """)

    def cleanup(self) -> None:
        pass


class PostgresCopyTable(BaseOperator):
    """
    The operator uses `DuckDB <https://duckdb.org/>`_ to connect directly to
    both the source and target PostgreSQL databases.

    It then copies the data in a single, efficient step using a parquet file
    as an intermediate storage. This method avoids loading the data
    into the Airflow worker's memory, making the transfer significantly faster for large tables.

    This operator requires the following Python packages:

    * ``apache-airflow-providers-postgres``
    * ``duckdb``

    The operator will automatically attempt to install and load the ``postgres`` extension for DuckDB at runtime.

    :param source_conn_id: The Airflow connection ID for the source PostgreSQL database.
    :type source_conn_id: str
    :param target_conn_id: The Airflow connection ID for the target PostgreSQL database.
    :type target_conn_id: str
    :param source_table: The name of the table to copy from the source database.
    :type source_table: str
    :param target_table: The name of the table to be created or replaced in the target database.
    :type target_table: str
    :param target_schema: The name of the schema to be used on the target connection.
    :type target_schema: str
    :param source_schema: The name of the schema to be used on the source connection.
    :type source_schema: str
    :param if_exists: Defines how to handle the target table if it already exists.
        Can be 'replace' (drops and recreates the table) or 'truncate' (empties
        the existing table and inserts new data). Defaults to 'replace'.
    :type if_exists: str
    :param exclude_columns: A list of column names to exclude from the copy operation.
        These columns will not be transferred from the source table to the target table.
        Defaults to an empty list, meaning all columns are copied.
    :type exclude_columns: List[str]
    :param aws_conn_id: The Airflow connection ID for AWS S3. Optional - if not provided,
        local storage will be used instead.
    :type aws_conn_id: Optional[str]
    :param bucket_uri: S3 bucket URI (e.g., 's3://my-bucket/path'). Only required if aws_conn_id is provided.
    :type bucket_uri: Optional[str]
    :param local_storage_path: Local directory path for storing intermediate parquet files.
        Defaults to '/tmp/airflow_postgres_copy'. Only used when aws_conn_id is not provided.
    :type local_storage_path: str
    """

    template_fields = (
        'source_conn_id',
        'target_conn_id',
        'source_table',
        'target_table',
        'source_schema',
        'target_schema',
        'if_exists',
        'exclude_columns',
        'aws_conn_id',
        'bucket_uri',
        'local_storage_path',
    )

    _SOURCE_DB = 'source_db'
    _TARGET_DB = 'target_db'
    _VALID_IF_EXISTS = ('replace', 'truncate')

    def __init__(
        self,
        source_conn_id: str,
        target_conn_id: str,
        source_table: str,
        target_table: str,
        aws_conn_id: Optional[str] = None,
        bucket_uri: Optional[str] = None,
        local_storage_path: str = '/tmp/airflow_postgres_copy',
        source_schema: str = 'public',
        target_schema: str = 'public',
        if_exists: Literal[*_VALID_IF_EXISTS] = 'replace',
        exclude_columns: Optional[List[str]] = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.source_conn_id = source_conn_id
        self.target_conn_id = target_conn_id
        self.source_table = source_table
        self.target_table = target_table
        self.source_schema = source_schema
        self.target_schema = target_schema
        self.if_exists = if_exists
        self.exclude_columns = exclude_columns or []
        self.aws_conn_id = aws_conn_id
        self.bucket_uri = bucket_uri
        self.local_storage_path = local_storage_path

    def execute(self, context):
        if self.if_exists not in self._VALID_IF_EXISTS:
            raise AirflowException(
                f"Invalid value for if_exists: '{self.if_exists}'. Must be one of {self._VALID_IF_EXISTS}."
            )

        storage = self._get_storage()
        source_hook = PostgresHook(postgres_conn_id=self.source_conn_id)
        target_hook = PostgresHook(postgres_conn_id=self.target_conn_id)
        source_query = self._build_source_query()

        try:
            with duckdb.connect() as duckdb_conn:
                duckdb_conn.execute('INSTALL postgres; LOAD postgres;')

                self._extract(duckdb_conn, source_hook, source_query, storage)
                self._load(duckdb_conn, target_hook, storage.filepath)
        finally:
            storage.cleanup()

    def _extract(
        self,
        conn: duckdb.DuckDBPyConnection,
        source_hook: PostgresHook,
        source_query: str,
        storage: TempStorage,
    ) -> None:
        self._attach_database(conn, source_hook.get_uri(), self.source_schema, self._SOURCE_DB)
        storage.setup(conn)
        storage.copy(conn, source_query)
        self._detach_database(conn, self._SOURCE_DB)

    def _load(
        self,
        conn: duckdb.DuckDBPyConnection,
        target_hook: PostgresHook,
        storage_filepath: str,
    ) -> None:
        self._attach_database(conn, target_hook.get_uri(), self.target_schema, self._TARGET_DB)

        target_table_path = f'{self._TARGET_DB}.{self.target_table}'
        self._get_strategy().write(conn, target_table_path, storage_filepath)

        self._detach_database(conn, self._TARGET_DB)

    def _get_storage(self) -> TempStorage:
        if self.aws_conn_id:
            if not self.bucket_uri:
                raise AirflowException("bucket_uri is required when aws_conn_id is provided")
            return S3TempStorage(self.bucket_uri, self.source_table, self.aws_conn_id, self.log)
        return LocalTempStorage(self.local_storage_path, self.source_table, self.log)

    def _get_strategy(self) -> WriteStrategy:
        if self.if_exists == 'replace':
            return ReplaceStrategy(self.log)
        return TruncateStrategy(self.log, self._sync_table_schema)

    def _build_source_query(self) -> str:
        if self.exclude_columns:
            columns = ','.join(self.exclude_columns)
            return f'SELECT * EXCLUDE({columns}) FROM {self._SOURCE_DB}.{self.source_table}'

        return f'SELECT * FROM {self._SOURCE_DB}.{self.source_table}'

    def _sync_table_schema(
        self,
        conn: duckdb.DuckDBPyConnection,
        target_table: str,
        storage_filepath: str,
    ) -> None:
        source_cols_desc = conn.execute(f"DESCRIBE '{storage_filepath}'").fetchall()
        target_cols_desc = conn.execute(f"DESCRIBE {target_table}").fetchall()

        source_columns = {col[0]: col[1] for col in source_cols_desc}
        target_columns = {col[0] for col in target_cols_desc}

        columns_to_add = {
            k: v for k, v in source_columns.items() if k not in target_columns
        }

        for col_name, col_type in columns_to_add.items():
            self.log.info(f"Adding column '{col_name}' with type '{col_type}' to table '{target_table}'")
            conn.execute(f'ALTER TABLE {target_table} ADD COLUMN "{col_name}" {col_type}')

    def _attach_database(
        self,
        conn: duckdb.DuckDBPyConnection,
        db_uri: str,
        db_schema: str,
        loaded_db_name: str,
    ) -> None:
        conn.execute(f"""
            ATTACH '{db_uri}' AS
            {loaded_db_name} (TYPE postgres, SCHEMA '{db_schema}')
        """)

    def _detach_database(self, conn: duckdb.DuckDBPyConnection, loaded_db_name: str) -> None:
        conn.execute(f'DETACH DATABASE {loaded_db_name}')
