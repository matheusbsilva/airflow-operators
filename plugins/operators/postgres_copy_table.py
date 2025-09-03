from typing import List

import duckdb

from airflow.models.baseoperator import BaseOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook


class PostgresCopyTable(BaseOperator):
    """
    The operator uses `DuckDB <https://duckdb.org/>`_ to connect directly to
    both the source and target PostgreSQL databases.

    It then copies the data in a single, efficient step using a
    ``CREATE TABLE AS SELECT`` command. This method avoids loading the data
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
    """
    template_fields = ('source_conn_id', 'target_conn_id', 'exclude_columns',
                       'source_table', 'target_table', 'if_exists')

    def __init__(
        self,
        source_conn_id: str,
        target_conn_id: str,
        source_table: str,
        target_table: str,
        source_schema: str = 'public',
        target_schema: str = 'public',
        if_exists: str = 'replace',
        exclude_columns: List[str] = [],
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.source_conn_id = source_conn_id
        self.target_conn_id = target_conn_id
        self.source_db_name = 'source'
        self.target_db_name = 'target'
        self.source_table = f'{self.source_db_name}.{source_table}'
        self.target_table = f'{self.target_db_name}.{target_table}'
        self.source_schema = source_schema
        self.target_schema = target_schema
        self.if_exists = if_exists
        self.exclude_columns = exclude_columns

        self.source_hook = self.load_conn(source_conn_id)
        self.target_hook = self.load_conn(target_conn_id)

    def execute(self, context):
        with duckdb.connect() as duckdb_conn:
            duckdb_conn.execute('INSTALL postgres; LOAD postgres;')
            source_query = self.build_source_query()

            self.attach_databases(duckdb_conn)

            if self.if_exists == 'replace':
                self.log.info(f"Starting copy from '{self.source_table}' to '{self.target_table}'")

                duckdb_conn.execute(f"""
                    CREATE OR REPLACE TABLE {self.target_table} AS
                    {source_query}
                """)

            if self.if_exists == 'truncate':
                try:
                    source_cols_desc = duckdb_conn.execute(f"DESCRIBE {self.source_table}").fetchall()
                    target_cols_desc = duckdb_conn.execute(f"DESCRIBE {self.target_table}").fetchall()

                    source_columns = {col[0]: col[1] for col in source_cols_desc}
                    target_columns = {col[0] for col in target_cols_desc}

                    columns_to_add = {
                        k: v for k, v in source_columns.items() if k not in target_columns
                    }

                    for col_name, col_type in columns_to_add.items():
                        self.log.info(f"Adding column '{col_name}' with type '{col_type}' to table '{self.target_table}'")
                        duckdb_conn.execute(f'ALTER TABLE {self.target_table} ADD COLUMN "{col_name}" {col_type}')

                    self.log.info(f"Truncating table {self.target_table}")

                    duckdb_conn.execute(f'TRUNCATE {self.target_table}')
                except duckdb.CatalogException as err:
                    self.log.warn(err)

                    self.log.info(f'Creating table {self.target_table}')
                    duckdb_conn.execute(f"""
                        CREATE TABLE {self.target_table} AS
                        {source_query}
                        WITH NO DATA
                    """)

                self.log.info(f"Inserting values from {self.source_table} to {self.target_table}")
                duckdb_conn.execute(f"""
                    INSERT INTO {self.target_table} BY NAME (
                        {source_query}
                    )
                """)

    def build_source_query(self) -> str:
        if self.exclude_columns:
            columns = ','.join(self.exclude_columns)
            return f'SELECT * EXCLUDE({columns}) FROM {self.source_table}'

        return f'SELECT * FROM {self.source_table}'

    def attach_databases(self, conn: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyConnection:
        conn.execute(f"""
            ATTACH '{self.source_hook.get_uri()}' AS
            {self.source_db_name} (TYPE postgres, SCHEMA '{self.source_schema}')
        """)
        conn.execute(f"""
            ATTACH '{self.target_hook.get_uri()}' AS
            {self.target_db_name} (TYPE postgres, SCHEMA '{self.target_schema}')
        """)

        return conn


    def load_conn(self, conn_id) -> PostgresHook:
        hook = PostgresHook(postgres_conn_id=conn_id)

        self.log.info(f'Loaded connection: {conn_id}')

        return hook