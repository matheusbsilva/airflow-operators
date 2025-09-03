# Custom Airflow Operators

This repository contains a collection of custom operators for Apache Airflow.

## Installation

To use these operators, copy the `plugins` directory into your Airflow project's root directory.

## Available Operators

This repository includes the following operators:

*   **BigQueryToPostgresql**: Transfers data from BigQuery to PostgreSQL. For more details, see the operator file at [`plugins/operators/bigquery_to_postgresql.py`](plugins/operators/bigquery_to_postgresql.py).
*   **EMRPackageDeploy**: Packages a Python project and deploys it to S3 for use with Amazon EMR. For more details, see the operator file at [`plugins/operators/emr_package_deploy.py`](plugins/operators/emr_package_deploy.py).
*   **PostgresCopyTable**: Efficiently copies a table from one PostgreSQL database to another. For more details, see the operator file at [`plugins/operators/postgres_copy_table.py`](plugins/operators/postgres_copy_table.py).

For usage examples, please refer to the docstrings and comments within each operator's source file.