import boto3
import os
import pickle
from src.exception import RentException
import sys


class SimpleStorageService:
    def __init__(self):
        try:
            self.s3_client = boto3.client('s3')
        except Exception as e:
            raise RentException(e, sys) from e

    def s3_key_path_available(self, bucket_name: str, s3_key: str) -> bool:
        """
        Check if file exists in S3 bucket
        """
        try:
            self.s3_client.head_object(Bucket=bucket_name, Key=s3_key)
            return True
        except Exception:
            return False

    def upload_file(self, from_filename: str, to_filename: str, bucket_name: str, remove: bool = False):
        """
        Upload file to S3
        """
        try:
            self.s3_client.upload_file(from_filename, bucket_name, to_filename)

            if remove:
                os.remove(from_filename)

        except Exception as e:
            raise RentException(e, sys) from e

    def load_model(self, model_name: str, bucket_name: str):
        """
        Download and load model from S3
        """
        try:
            local_file = model_name

            # download file
            self.s3_client.download_file(bucket_name, local_file, local_file)

            # load model
            with open(local_file, "rb") as f:
                model = pickle.load(f)

            return model

        except Exception as e:
            raise RentException(e, sys) from e