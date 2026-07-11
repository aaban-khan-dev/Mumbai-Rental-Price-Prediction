import os
import sys
import pandas as pd

from src.constant import *
from src.exception import RentException
from src.logger import logging
from src.configuration.mongo_db_connection import MongoDBclient


class MongoUploader:
    """
    Uploads the scraped rental CSV into MongoDB.
    Uploads ONLY if the collection is empty, so running the pipeline a second
    time does not create duplicate data.
    """

    def __init__(self,
                 database_name: str = MONGO_DATABASE_NAME,
                 collection_name: str = MONGO_COLLECTION_NAME):
        try:
            self.database_name = database_name
            self.collection_name = collection_name
            # reuse the shared Mongo client from configuration
            self.mongo_client = MongoDBclient(database_name=database_name)
            self.collection = self.mongo_client.database[collection_name]
        except Exception as e:
            raise RentException(e, sys) from e

    def upload_if_empty(self, csv_file_path: str = DATA_FILE_PATH):
        """
        Check the collection. If it already has data, skip the upload.
        If it is empty, read the CSV and insert all rows.
        """
        try:
            existing_count = self.collection.count_documents({})
            logging.info(f"Collection '{self.collection_name}' has {existing_count} documents")

            # ---- Redundancy guard: data already present -> do not upload again ----
            if existing_count > 0:
                logging.info("Data already present in MongoDB. Skipping upload.")
                return

            # ---- Collection is empty -> read the CSV and upload ----
            logging.info(f"Reading dataset from {csv_file_path}")
            df = pd.read_csv(csv_file_path)
            logging.info(f"Dataset shape: {df.shape}")

            records = df.to_dict(orient="records")
            self.collection.insert_many(records)
            logging.info(f"Uploaded {len(records)} records to '{self.collection_name}'")
            
        except Exception as e:
            raise RentException(e, sys) from e
