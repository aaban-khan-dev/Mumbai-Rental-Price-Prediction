from datetime import datetime
import os


AWS_S3_BUCKET_NAME = "mumbai-rental-prediction"
MONGO_DATABASE_NAME = "Mumbai_Rent_Prediction"
MONGO_COLLECTION_NAME = "rentals"

DATA_FILE_PATH = "data/magicbricks_rentals.csv"
TARGET_COLUMN = "rent"

MODEL_FILE_NAME = "model"
MODEL_FILE_EXTENSION = ".pkl"

artifact_folder_name = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
artifact_folder =  os.path.join("artifacts", artifact_folder_name)

REGION_NAME = os.getenv("REGION_NAME")