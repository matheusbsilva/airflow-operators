from typing import List, Optional
import os
from pathlib import Path

import duckdb

from airflow.models.baseoperator import BaseOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.amazon.aws.hooks.s3 import S3Hook


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
    template_fields = ('aws_conn_id', 'source_conn_id', 'target_conn_id', 'exclude_columns',
                       'source_table', 'target_table', 'if_exists', 'bucket_uri', 'local_storage_path')

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
        if_exists: str = 'replace',
        exclude_columns: List[str] = [],
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.source_conn_id = source_conn_id
        self.target_conn_id = target_conn_id
        self.loaded_source_db = f'source_db'
        self.loaded_target_db = f'target_db'
        self.source_table = source_table
        self.target_table = target_table
        self.source_schema = source_schema
        self.target_schema = target_schema
        self.if_exists = if_exists
        self.exclude_columns = exclude_columns
        self.aws_conn_id = aws_conn_id
        self.bucket_uri = bucket_uri
        self.local_storage_path = local_storage_path

        # Determine storage mode
        self.use_s3 = aws_conn_id is not None

        if self.use_s3:
            if not bucket_uri:
                raise ValueError("bucket_uri is required when aws_conn_id is provided")
            self.storage_filepath = f'{self.bucket_uri}/{self.source_table}.parquet'
            self.s3_hook = self.load_s3_conn(aws_conn_id)
        else:
            # Use local storage
            Path(self.local_storage_path).mkdir(parents=True, exist_ok=True)
            self.storage_filepath = os.path.join(self.local_storage_path, f'{self.source_table}.parquet')

        self.source_hook = self.load_postgres_conn(source_conn_id)
        self.target_hook = self.load_postgres_conn(target_conn_id)
        self.source_query = self.build_source_query()

    def execute(self, context):
        with duckdb.connect() as duckdb_conn:
            duckdb_conn.execute('INSTALL postgres; LOAD postgres;')

            self.extract(duckdb_conn)

            self.load(duckdb_conn)

            # Cleanup local file if using local storage
            if not self.use_s3:
                self.cleanup_local_file()

    def extract(self, conn: duckdb.DuckDBPyConnection):
        self.attach_database(
            conn,
            self.source_hook.get_uri(),
            self.source_schema,
            self.loaded_source_db,
        )

        self.copy_source_to_storage(conn)

        self.detach_database(conn, self.loaded_source_db)

    def load(self, conn: duckdb.DuckDBPyConnection):
        self.attach_database(
            conn,
            self.target_hook.get_uri(),
            self.target_schema,
            self.loaded_target_db
        )

        self.create_table(conn)

        self.log.info(f"Inserting values from {self.storage_filepath} to {self.target_table}")

        conn.execute(f"""
            INSERT INTO {self.loaded_target_db}.{self.target_table} BY NAME (
                SELECT * FROM '{self.storage_filepath}'
            )
        """)

        self.detach_database(conn, self.loaded_target_db)

    def create_table(self, conn: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyConnection:
        target_table_path = f'{self.loaded_target_db}.{self.target_table}'

        if self.if_exists == 'replace':
            conn.execute(f"""
                CREATE OR REPLACE TABLE {target_table_path} AS
                SELECT * FROM '{self.storage_filepath}'
                WITH NO DATA
            """)

        if self.if_exists == 'truncate':
            try:
                self.sync_table_schema(conn, target_table_path)

                self.log.info(f"Truncating table {target_table_path}")

                conn.execute(f'TRUNCATE {target_table_path}')
            except duckdb.CatalogException as err:
                self.log.warn(err)

                self.log.info(f'Creating table {target_table_path}')

                conn.execute(f"""
                    CREATE TABLE {target_table_path} AS
                    SELECT * FROM '{self.storage_filepath}'
                    WITH NO DATA
                """)

    def copy_source_to_storage(self, conn: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyConnection:
        if self.use_s3:
            self.attach_s3(conn)

        storage_type = "S3" if self.use_s3 else "local storage"
        self.log.info(f"Copying data from {self.source_table} to {storage_type}: {self.storage_filepath}")

        conn.execute(f"""
            COPY ({self.source_query}) TO '{self.storage_filepath}' (
                FORMAT parquet,
                COMPRESSION zstd
            )
        """)

        return conn

    def build_source_query(self) -> str:
        if self.exclude_columns:
            columns = ','.join(self.exclude_columns)

            return f'SELECT * EXCLUDE({columns}) FROM {self.loaded_source_db}.{self.source_table}'

        return f'SELECT * FROM {self.loaded_source_db}.{self.source_table}'

    def sync_table_schema(self, conn: duckdb.DuckDBPyConnection, target_table: str) -> duckdb.DuckDBPyConnection:
        source_cols_desc = conn.execute(f"DESCRIBE '{self.storage_filepath}'").fetchall()
        target_cols_desc = conn.execute(f"DESCRIBE {target_table}").fetchall()

        source_columns = {col[0]: col[1] for col in source_cols_desc}
        target_columns = {col[0] for col in target_cols_desc}

        columns_to_add = {
            k: v for k, v in source_columns.items() if k not in target_columns
        }

        for col_name, col_type in columns_to_add.items():
            self.log.info(f"Adding column '{col_name}' with type '{col_type}' to table '{target_table}'")

            conn.execute(f'ALTER TABLE {target_table} ADD COLUMN "{col_name}" {col_type}')

        return conn

    def attach_database(
        self,
        conn: duckdb.DuckDBPyConnection,
        db_uri: str,
        db_schema: str,
        loaded_db_name: str,
    ) -> duckdb.DuckDBPyConnection:
        conn.execute(f"""
            ATTACH '{db_uri}' AS
            {loaded_db_name} (TYPE postgres, SCHEMA '{db_schema}')
        """)

        return conn

    def detach_database(self, conn: duckdb.DuckDBPyConnection, loaded_db_name: str):
        conn.execute(f"""
            DETACH DATABASE {loaded_db_name}
        """)

        return conn

    def attach_s3(self, conn: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyConnection:

        conn.execute('INSTALL httpfs; LOAD httpfs;')

        region_name = self.s3_hook.region_name
        credentials = self.s3_hook.get_credentials()

        conn.execute(f"""
            CREATE SECRET (
                TYPE s3,
                KEY_ID '{credentials.access_key}',
                SECRET '{credentials.secret_key}',
                REGION '{region_name}'
            );
        """)

        return conn

    def cleanup_local_file(self):
        """Remove the local parquet file after successful load."""
        try:
            if os.path.exists(self.storage_filepath):
                os.remove(self.storage_filepath)
                self.log.info(f"Cleaned up local file: {self.storage_filepath}")
        except Exception as e:
            self.log.warning(f"Failed to cleanup local file {self.storage_filepath}: {e}")

    def load_postgres_conn(self, conn_id: str) -> PostgresHook:
        hook = PostgresHook(postgres_conn_id=conn_id)

        return hook

    def load_s3_conn(self, conn_id: str) -> S3Hook:
        hook = S3Hook(aws_conn_id=conn_id)

        return hook