import os
import sys

import certifi
import pymongo

from src.constant import *
from src.exception import RentException   

ca = certifi.where()

class MongoDBclient:
    client = None

    def __init__(self, database_name=MONGO_DATABASE_NAME):
        try:
            if MongoDBclient.client is None:
                mongo_db_url = os.getenv("MONGO_DB_URL")
                if mongo_db_url is None:
                    raise RentException("MongoDB URL is not set in environment variables.", sys)
                MongoDBclient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
            self.client = MongoDBclient.client
            self.database = self.client[database_name]
            self.database_name = database_name
        except Exception as e:
            raise RentException(e, sys) from e
