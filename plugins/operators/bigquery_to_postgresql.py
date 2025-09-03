from typing import Optional

from google.cloud import bigquery

from airflow.models.baseoperator import BaseOperator
from airflow.providers.google.cloud.hooks.bigquery import BigQueryHook
from airflow.providers.postgres.hooks.postgres import PostgresHook


class BigQueryToPostgresql(BaseOperator):
    """
    Transer data from BigQuery to Postgresql with custom queries.

    :param gcp_conn_id: GCP connection id.
    :param postgres_conn_id: Postgresql connection id.
    :param sql: SQL query to execute on BigQuery.
    :param if_exists: Behavior when target table exists,
        the available values are: 'replace' and 'append'.
    :param query_parameters: Parameters to SQL query.
        It must be a list of bigquery parameter objects,
        more info on https://cloud.google.com/bigquery/docs/parameterized-queries.
    :param target_table: Postgres target table.
    :param target_schema: Postgres target schema.
    :param location: Bigquery project location.
    :param project_id: Bigquery project id.
    :param page_size: How many rows to load in memory for each interation on queried data.
    """

    template_fields = (
        'sql',
        'target_table',
        'if_exists',
        'location',
        'project_id',
        'target_table',
        'page_size',
        'postgres_conn_id',
        'gcp_conn_id',
        'query_parameters',
        'target_schema'
    )

    def __init__(
        self,
        gcp_conn_id: str,
        postgres_conn_id: str,
        sql: str,
        target_table: str,
        if_exists: str,
        target_schema: Optional[str] = None,
        location: Optional[str] = None,
        project_id: Optional[str] = None,
        query_parameters: Optional[list] = None,
        page_size: Optional[int] = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.gcp_conn_id = gcp_conn_id
        self.postgres_conn_id = postgres_conn_id
        self.sql = sql
        self.location = location
        self.project_id = project_id
        self.target_table = target_table
        self.if_exists = if_exists
        self.query_parameters = query_parameters
        self.page_size = page_size
        self.target_schema = target_schema
        self.job_config = None

    def build_query_parameters(self):
        params = []

        for item in self.query_parameters:
            params.append(
                bigquery.ScalarQueryParameter(
                    item['name'],
                    item['type'],
                    item['value']
                )
            )

        return params

    def execute(self, context):
        big_query_hook = BigQueryHook(
            gcp_conn_id=self.gcp_conn_id,
        )
        big_query_client = big_query_hook.get_client(
            self.project_id,
            self.location
        )

        if self.query_parameters:
            self.job_config = bigquery.QueryJobConfig(
                query_parameters=self.build_query_parameters()
            )

        self.log.info(self.sql)
        self.log.info(self.job_config.to_api_repr())
        query_job = big_query_client.query(
            self.sql,
            job_config=self.job_config
        )

        self.log.info(self.if_exists)

        postgres_hook = PostgresHook(postgres_conn_id=self.postgres_conn_id)
        postgres_engine = postgres_hook.get_sqlalchemy_engine()

        result = query_job.result(page_size=self.page_size)

        with postgres_engine.begin() as conn:
            if self.if_exists == 'replace':
                table_name = (
                    f'{self.target_schema}.{self.target_table}'
                    if self.target_schema
                    else self.target_table
                )
                self.log.info(f'Dropping table {table_name}')
                conn.execute(f"DROP TABLE IF EXISTS {table_name}")

            for df in result.to_dataframe_iterable():
                self.log.info(f"Loading result, with shape: {df.shape}")
                df.to_sql(
                    self.target_table,
                    conn,
                    schema=self.target_schema,
                    if_exists='append',
                    index=False
                )
