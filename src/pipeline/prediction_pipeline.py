import sys
import os
import numpy as np
import pandas as pd

from src.constant import *
from src.exception import RentException
from src.logger import logging
from src.utils.main_utils import MainUtils
from src.model.s3_estimator import RentEstimator


class PredictionPipeline:
    """
    Serves rent predictions from the trained model.

    - Loads the model ONCE from S3 (cached in memory) and reuses it for every request.
    - Reads prediction_schema.yaml to apply defaults for optional fields and to
      derive the engineered features (floor_ratio, has_metro) the user does not supply.
    """

    # class-level cache so the model is loaded only once across requests
    _cached_model = None

    def __init__(self):
        try:
            self.utils = MainUtils()
            self.model_file_name = MODEL_FILE_NAME + MODEL_FILE_EXTENSION   # 'model.pkl'
            self.prediction_schema = self.utils.read_yaml_file(
                os.path.join("config", "prediction_schema.yaml")
            )
        except Exception as e:
            raise RentException(e, sys) from e

    #  load model once, then cache 
    def get_model(self):
        """Load the model from S3 the first time, then reuse the cached copy."""
        try:
            if PredictionPipeline._cached_model is None:
                logging.info("Loading model from S3 (first request)")
                estimator = RentEstimator(
                    bucket_name=AWS_S3_BUCKET_NAME,
                    model_name=self.model_file_name,
                )
                PredictionPipeline._cached_model = estimator.load_model()
                logging.info("Model loaded and cached in memory")
            return PredictionPipeline._cached_model
        except Exception as e:
            raise RentException(e, sys) from e

    # apply defaults + derive features
    def prepare_input(self, user_input: dict) -> pd.DataFrame:
        """
        Take the raw user input (dict from the UI/API), fill defaults for any
        optional field the user didn't provide, and derive floor_ratio & has_metro.
        Returns a single-row DataFrame in the structure the preprocessor expects.
        """
        try:
            data = dict(user_input)  # copy so we don't mutate the caller's dict

            #  fill defaults for optional fields not supplied by the user 
            defaults = self.prediction_schema.get("defaults", {})
            for field, default_value in defaults.items():
                if field not in data or data[field] is None or data[field] == "":
                    data[field] = default_value

            # derive features the user never provides 
            # floor_ratio = floor_num / total_floors
            floor_num = float(data.get("floor_num", 0))
            total_floors = float(data.get("total_floors", 0))
            data["floor_ratio"] = (floor_num / total_floors) if total_floors > 0 else 0.0

            # has_metro: 1 if the user indicated a metro time, else 0
            metro_mins = data.get("metro_mins")
            if metro_mins is None or metro_mins == "" or float(metro_mins) <= 0:
                data["has_metro"] = 0
                # use a 'far' sentinel so the model reads "no metro nearby"
                data["metro_mins"] = 60
            else:
                data["has_metro"] = 1
                data["metro_mins"] = float(metro_mins)

            # build the DataFrame in the exact feature order the model expects 
            feature_order = [
                "property_type", "locality", "furnishing", "facing",
                "bhk", "carpet_area", "floor_num", "total_floors", "floor_ratio",
                "available_immediately",
                "overlooks_garden", "overlooks_pool", "overlooks_main_road",
                "metro_mins", "has_metro",
                "near_school", "near_hospital", "near_mall", "near_bus", "near_railway",
            ]
            row = {col: data.get(col) for col in feature_order}
            df = pd.DataFrame([row])
            logging.info(f"Prepared input row: {row}")
            return df

        except Exception as e:
            raise RentException(e, sys) from e

    # predict 
    def predict(self, user_input: dict) -> float:
        """Return the predicted monthly rent (in rupees) for one user input."""
        try:
            model = self.get_model()
            input_df = self.prepare_input(user_input)
            prediction = model.predict(input_df)          # RentModel already returns rupees
            predicted_rent = float(np.ravel(prediction)[0])
            logging.info(f"Predicted rent: {predicted_rent:.0f}")
            return round(predicted_rent, 2)
        except Exception as e:
            raise RentException(e, sys) from e
