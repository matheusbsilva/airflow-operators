import os
import shutil
import zipfile
from pathlib import Path
from typing import List, Optional

from airflow.models.baseoperator import BaseOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook


def find_files(directory, excluded_dirs=[], search=None) -> List[str]:
    files = []
    for root, dirs, filenames in os.walk(directory):

        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        for filename in filenames:
            if search is None or filename == search or filename.endswith(search):
                files.append(os.path.join(root, filename))
    return files


class EMRPackageDeploy(BaseOperator):
    """
    Package and deploy an EMR python project to S3.
    All local dependencies of entrypoint will be packaged in a zip file.


    :param path: Python project root path.
    :param entrypoint: Project entrypoint script.
    :param s3_path: Full S3 URI of project destination path on S3.
    :param aws_conn_id: AWS Connection ID.
    """

    template_fields = ("path", "entrypoint", "s3_path", "aws_conn_id")

    def __init__(
        self,
        path: str,
        entrypoint: str,
        s3_path: str,
        aws_conn_id: Optional[str] = None,
        **kwargs
    ) -> None:

        super().__init__(**kwargs)
        self.path = Path(path)
        self._path = path
        self.entrypoint = entrypoint
        self.entrypoint_path = self.path / entrypoint
        self.s3_path = s3_path
        self.dist_dir = self.path / 'dist'
        self.zip_name = 'pyfiles.zip'
        self.zip_path = self.dist_dir / self.zip_name
        self.aws_conn_id = aws_conn_id

    def package(self):
        self.log.info(f'Dist directory: {self.dist_dir.as_posix()}')
        files = find_files(self._path, ['.venv', 'dist', '__pycache__'])

        files.remove(self.entrypoint_path.as_posix())

        self.dist_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(self.dist_dir / self.zip_path, 'w') as zf:
            for file in files:
                self.log.info(f"Packaging file {file}")
                job_basepath = os.path.commonpath([self.entrypoint_path, file])
                relpath = os.path.relpath(file, job_basepath)
                zf.write(file, relpath)

        self.log.info(f'Packaging complete, generated: {self.dist_dir}')

    def deploy(self):
        bucket, prefix = S3Hook.parse_s3_url(self.s3_path)
        bucket_client = S3Hook(aws_conn_id=self.aws_conn_id).get_bucket(bucket)

        self.log.info(f"{bucket}, {prefix}")

        src_target = {
            self.entrypoint_path: os.path.join(prefix, self.entrypoint),
            self.zip_path:  os.path.join(prefix, self.zip_name)
        }

        for src, target in src_target.items():
            bucket_client.upload_file(src.as_posix(), target)

        self.log.info(f"Uploaded {self.entrypoint} and local python modules to {self.s3_path}")

        shutil.rmtree(self.dist_dir)

        self.log.info(f"Distribution directory {self.dist_dir.as_posix()} cleaned up.")

    def execute(self, context):
        self.package()
        self.deploy()