import sys
import os

from src.data_access.mongo_uploader import MongoUploader           
from src.component.data_ingestion import DataIngestion
from src.component.data_validation import DataValidation
from src.component.data_transformation import DataTransformation
from src.component.model_trainer import ModelTrainer
from src.model.s3_estimator import RentEstimator                  
from src.constant import *
from src.exception import RentException
from src.logger import logging


class TrainingPipeline:

    #  upload data to MongoDB if the collection is empty 
    def start_data_upload(self):
        try:
            logging.info("Starting data upload to MongoDB")
            mongo_uploader = MongoUploader()
            mongo_uploader.upload_if_empty(csv_file_path=DATA_FILE_PATH)
            logging.info("Data upload step completed")
        except Exception as e:
            raise RentException(e, sys) from e

    def start_data_ingestion(self):
        try:
            data_ingestion = DataIngestion()
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise RentException(e, sys) from e

    def start_data_validation(self, raw_data_dir):
        try:
            data_validation = DataValidation(raw_data_store_dir=raw_data_dir)
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise RentException(e, sys) from e

    def start_data_transformation(self, validation_data_dir):
        try:
            data_transformation = DataTransformation(valid_data_dir=validation_data_dir)
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation()
            return train_arr, test_arr, preprocessor_path
        except Exception as e:
            raise RentException(e, sys) from e

    def start_model_trainer(self, train_arr, test_arr, preprocessor_path):
        try:
            model_trainer = ModelTrainer()
            model_score, trained_model_path = model_trainer.initiate_model_trainer(
                train_arr, test_arr, preprocessor_path
            )
            return model_score, trained_model_path
        except Exception as e:
            raise RentException(e, sys) from e

    #  push the trained model to the S3 bucket 
    def start_model_pusher(self, trained_model_path):
        try:
            logging.info("Pushing trained model to S3 bucket")
            model_file_name = MODEL_FILE_NAME + MODEL_FILE_EXTENSION   
            rent_estimator = RentEstimator(
                bucket_name=AWS_S3_BUCKET_NAME,
                model_name=model_file_name,
            )
            rent_estimator.save_model(from_file=trained_model_path, remove=False)
            logging.info(f"Model pushed to S3 bucket '{AWS_S3_BUCKET_NAME}' as '{model_file_name}'")
        except Exception as e:
            raise RentException(e, sys) from e

    def run_pipeline(self):
        try:
            # ensure data is in MongoDB before ingestion
            self.start_data_upload()

            raw_data_dir = self.start_data_ingestion()
            validation_data_dir = self.start_data_validation(raw_data_dir)
            train_arr, test_arr, preprocessor_path = self.start_data_transformation(validation_data_dir)
            model_score, trained_model_path = self.start_model_trainer(
                train_arr, test_arr, preprocessor_path
            )

            # NEW: push the final model to S3
            self.start_model_pusher(trained_model_path)

            logging.info(f"Training completed. Model R² score: {model_score}")
            print(f"Training completed. Model R² score: {model_score}")

        except Exception as e:
            raise RentException(e, sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
