import sys
from pandas import DataFrame

from src.exception import RentException
from src.cloud_storage.aws_storage import SimpleStorageService
from src.model.estimator import RentModel


class RentEstimator:
    """
    Save and retrieve the rent model in an S3 bucket, and use it for prediction.
    """

    def __init__(self, bucket_name, model_name):
        self.bucket_name = bucket_name
        self.model_name = model_name
        self.s3 = SimpleStorageService()
        self.loaded_model: RentModel = None

    def is_model_present(self, model_name):
        """Check whether the model file exists in the S3 bucket."""
        try:
            return self.s3.s3_key_path_available(self.bucket_name, model_name)
        except Exception as e:
            raise RentException(e, sys) from e

    def load_model(self) -> RentModel:
        """Download and load the model from S3."""
        try:
            return self.s3.load_model(self.model_name, self.bucket_name)
        except Exception as e:
            raise RentException(e, sys) from e

    def save_model(self, from_file, remove: bool = False) -> None:
        """
        Upload the local model file to S3.
        param from_file: local path of the model to upload
        param remove: if True, delete the local copy after upload
        """
        try:
            self.s3.upload_file(
                from_file,
                to_filename=self.model_name,
                bucket_name=self.bucket_name,
                remove=remove,
            )
        except Exception as e:
            raise RentException(e, sys) from e

    def predict(self, dataframe: DataFrame):
        """Load the model from S3 (once) and predict."""
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()
            return self.loaded_model.predict(dataframe)
        except Exception as e:
            raise RentException(e, sys) from e
