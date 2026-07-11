import os
import boto3

from src.constant import REGION_NAME


class S3Client:

    s3_client = None
    s3_resource = None

    def __init__(self, region_name=REGION_NAME):

        if S3Client.s3_client is None or S3Client.s3_resource is None:

            access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
            secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

            if access_key_id is None or secret_access_key is None:
                raise ValueError("AWS credentials are not set in environment variables.")

            S3Client.s3_resource = boto3.resource(
                service_name="s3",
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                region_name=region_name,
            )

            S3Client.s3_client = boto3.client(
                service_name="s3",
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                region_name=region_name,
            )

        self.s3_client = S3Client.s3_client
        self.s3_resource = S3Client.s3_resource