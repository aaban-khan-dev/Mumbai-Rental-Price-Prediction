from datetime import date
import sys
import os
from typing import List
import pandas as pd
import re
import shutil

from src.constant import *
from src.exception import RentException        
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass

sample_file_name = "rental_08012020_120000.csv"   
LENGTH_OF_DATE_STAMP = 8
LENGTH_OF_TIME_STAMP = 6

@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join("artifact_folder", "data_validation")
    valid_data_dir: str = os.path.join(data_validation_dir, "valid_data")
    invalid_data_dir: str = os.path.join(data_validation_dir, "invalid_data")


class DataValidation:
    def __init__(self, raw_data_store_dir: str):
        self.raw_data_store_dir = raw_data_store_dir
        self.data_validation_config = DataValidationConfig()
        self.main_utils = MainUtils()
        self._schema_config = self.main_utils.read_schema_config_file()

    #  FILE NAME VALIDATION
    def validate_file_name(self, file_path: str) -> bool:
        try:
            file_name = os.path.basename(file_path)

            pattern = r"^rental_\d{8}_\d{6}\.csv$"

            if re.match(pattern, file_name):
                return True
            logging.info(f"Invalid filename format: {file_name}")   
            return False

        except Exception as e:
            raise RentException(e, sys) from e

    #  COLUMN COUNT VALIDATION 
    def validate_number_of_columns(self, file_path: str) -> bool:
        try:
            df = pd.read_csv(file_path)

            expected_columns = [list(col.keys())[0] for col in self._schema_config["columns"]]
            actual_columns = list(df.columns)

            logging.info(f"Checking columns for: {file_path}")
            logging.info(f"Expected: {expected_columns}")
            logging.info(f"Actual: {actual_columns}")

            if actual_columns == expected_columns:
                return True
            logging.info("Column name mismatch")
            return False

        except Exception as e:
            raise RentException(e, sys) from e

    #  COLUMN TYPE VALIDATION 
    def validate_column_types(self, file_path: str) -> bool:
        """
        Check that each column's dtype loosely matches the type declared in schema.yaml.
        
        """
        try:
            df = pd.read_csv(file_path)

            #  also skip drop_columns here — a fully-null column (e.g.
            # description_raw) reads as float dtype, not object, so it would fail the
            # str type check even though it's valid raw data that gets dropped later.
            drop_columns = self._schema_config.get("drop_columns", []) or []

            # map schema type words to a simple pandas-kind check
            for col_dict in self._schema_config["columns"]:
                col_name = list(col_dict.keys())[0]
                expected_type = str(col_dict[col_name]).lower()

                if col_name not in df.columns:
                    continue  
                if col_name in drop_columns:
                    continue  

                actual_kind = df[col_name].dtype.kind  

                # str/object columns should read as object; float columns as float/int
                if expected_type in {"str", "object", "o"}:
                    ok = actual_kind == "O"
                elif expected_type in {"float", "float64"}:
                    ok = actual_kind in {"f", "i"}
                else:
                    ok = True  

                if not ok:
                    logging.info(
                        f"Type mismatch in '{col_name}': expected {expected_type}, "
                        f"got dtype-kind '{actual_kind}'"
                    )
                    return False

            return True

        except Exception as e:
            raise RentException(e, sys) from e

    #  MISSING VALUES VALIDATION 
    def validate_missing_values_in_whole_column(self, file_path: str) -> bool:
        try:
            df = pd.read_csv(file_path)

            #  skip columns that will be dropped in transformation.
            # Some raw scraped columns (e.g. description_raw) are fully null BY NATURE
            # and are removed later during cleaning. Checking them here would wrongly
            # mark every file INVALID. So we only validate columns we actually keep.
            drop_columns = self._schema_config.get("drop_columns", []) or []

            for column in df.columns:
                if column in drop_columns:
                    continue  # will be dropped in transformation
                if df[column].count() == 0:
                    logging.info(f"Column '{column}' has all missing values") 
                    return False

            return True

        except Exception as e:
            raise RentException(e, sys) from e

    def get_raw_batch_file_path(self) -> List:
        try:
            file_list = os.listdir(self.raw_data_store_dir)

            logging.info(f"Files found: {file_list}")

            return [os.path.join(self.raw_data_store_dir, file) for file in file_list]

        except Exception as e:
            raise RentException(e, sys) from e

    def move_raw_files_to_validation_dir(self, src_path: str, dst_path: str):
        try:
            os.makedirs(dst_path, exist_ok=True)

            if os.path.basename(src_path) not in os.listdir(dst_path):
                shutil.move(src_path, dst_path)

        except Exception as e:
            raise RentException(e, sys) from e

    def validate_raw_files(self) -> bool:
        try:
            raw_batch_file_path = self.get_raw_batch_file_path()

            validated_file_count = 0

            for raw_file in raw_batch_file_path:
                logging.info(f"Validating file: {raw_file}")

                file_name_status = self.validate_file_name(raw_file)
                column_status = self.validate_number_of_columns(raw_file)
                type_status = self.validate_column_types(raw_file)   # changed: new type check
                missing_value_status = self.validate_missing_values_in_whole_column(raw_file)

                logging.info(
                    f"Result -> Filename: {file_name_status}, Columns: {column_status}, "
                    f"Types: {type_status}, Missing: {missing_value_status}"
                )

                if file_name_status and column_status and type_status and missing_value_status:
                    validated_file_count += 1
                    logging.info("File is VALID")
                    self.move_raw_files_to_validation_dir(
                        raw_file, self.data_validation_config.valid_data_dir
                    )
                else:
                    logging.info("File is INVALID")
                    self.move_raw_files_to_validation_dir(
                        raw_file, self.data_validation_config.invalid_data_dir
                    )

            logging.info(f"Total valid files: {validated_file_count}")

            return validated_file_count > 0

        except Exception as e:
            raise RentException(e, sys) from e

    def initiate_data_validation(self):
        try:
            logging.info("Entered data validation")

            validation_status = self.validate_raw_files()

            if not validation_status:
                raise RentException("No data could be validated. Pipeline stopped", sys)

            valid_data_dir = self.data_validation_config.valid_data_dir
            logging.info(
                f"Data Validation completed. Valid files at: {valid_data_dir}"
            )

            logging.info("Exited data validation")
            return valid_data_dir

        except Exception as e:
            raise RentException(e, sys) from e
