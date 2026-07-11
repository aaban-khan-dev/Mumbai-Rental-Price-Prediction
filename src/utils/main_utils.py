import sys
from typing import Dict,Tuple
import os
import pandas as pd
import pickle
import yaml

from src.exception import RentException
from src.logger import logging
from src.constant import *

class MainUtils:
    
    def __init__(self):
        pass

    def read_yaml_file(self,filename:str)->Dict:
        try:
            with open(filename,"rb") as yaml_file:
                return yaml.safe_load(yaml_file)
            
        except Exception as e:
            raise RentException(e,sys) from e
        
    def read_schema_config_file(self)->Dict:
        try:
            return self.read_yaml_file(os.path.join("config","schema.yaml"))
        except Exception as e:
            raise RentException(e,sys) from e
        

    @staticmethod
    def save_object(file_path:str,obj:object)->None:
        logging.info("Entered the save_object method of MainUtils class")
        try:
            with open(file_path,"wb") as file_obj:
                pickle.dump(obj,file_obj)
            logging.info("Exited the save_object method of MainUtils class")
        except Exception as e:
            raise RentException(e,sys) from e
        
    @staticmethod
    def load_object(file_path:str)->object:
        logging.info("Entered the load_object method of MainUtils class")
        try:
            with open(file_path,"rb") as file_obj:
                return pickle.load(file_obj)
            logging.info("Exited the load_object method of MainUtils class")
        except Exception as e:
            raise RentException(e,sys) from e