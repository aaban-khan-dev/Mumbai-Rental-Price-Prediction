import subprocess
import sys

from src.logger import logging
from src.constant import REGION_NAME
from src.exception import RentException


class S3Sync:

    def sync_folder_to_S3(self, folder: str, aws_bucket_name: str):
        """
        Sync local folder to an S3 bucket.
        """
        try:
            logging.info(f"Uploading '{folder}' to S3 bucket '{aws_bucket_name}'")

            subprocess.run(
                [
                    "aws",
                    "s3",
                    "sync",
                    folder,
                    f"s3://{aws_bucket_name}",
                    "--region",
                    REGION_NAME,
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            logging.info("Folder uploaded successfully to S3.")

        except Exception as e:
            logging.error(f"Failed to upload folder to S3: {e}")
            raise RentException(e, sys) from e


    def sync_folder_from_S3(self, folder: str, aws_bucket_name: str):
        """
        Sync an S3 bucket to a local folder.
        """
        try:
            logging.info(f"Downloading S3 bucket '{aws_bucket_name}' to '{folder}'")

            subprocess.run(
                [
                    "aws",
                    "s3",
                    "sync",
                    f"s3://{aws_bucket_name}",
                    folder,
                    "--region",
                    REGION_NAME,
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            logging.info("Folder downloaded successfully from S3.")

        except Exception as e:
            logging.error(f"Failed to download folder from S3: {e}")
            raise RentException(e, sys) from e