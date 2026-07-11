import os
import sys
from typing import List

import numpy as np
import pandas as pd
from pymongo import MongoClient

from src.constant import *
from src.exception import RentException
from src.logger import logging            


class RentData:
    """
    This class helps to export the entire MongoDB collection as a pandas DataFrame.
    """

    def __init__(self, database_name: str = MONGO_DATABASE_NAME):
        try:
            self.database_name = database_name
            self.mongo_url = os.getenv("MONGO_DB_URL")
        except Exception as e:
            raise RentException(e, sys) from e

    def get_collection_names(self) -> List:
        mongo_db_client = MongoClient(self.mongo_url)
        return mongo_db_client[self.database_name].list_collection_names()

    def get_collection_data(self, collection_name: str) -> pd.DataFrame:
        try:
            mongo_db_client = MongoClient(self.mongo_url)
            collection = mongo_db_client[self.database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))
            logging.info(f"Fetched {df.shape[0]} rows from '{collection_name}'")

            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"])
            df = df.replace({"na": np.nan})
            return df
        except Exception as e:
            raise RentException(e, sys) from e

    def export_collections_as_dataframe(self):
        """
        Export every collection as a DataFrame.
        Yields (collection_name, DataFrame) pairs.
        """
        try:
            collections = self.get_collection_names()
            for collection_name in collections:
                df = self.get_collection_data(collection_name=collection_name)
                yield collection_name, df
        except Exception as e:
            raise RentException(e, sys) from e
