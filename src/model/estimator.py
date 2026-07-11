import sys
import numpy as np
from pandas import DataFrame

from src.exception import RentException
from src.logger import logging


class RentModel:
    """
    Wraps the preprocessor + trained model so a prediction is one clean call.
    The model is trained on log(rent), so predict() inverts it back to rupees.
    """

    def __init__(self, preprocessing_object, trainer_model_object: object):
        self.preprocessing_object = preprocessing_object
        self.trainer_model_object = trainer_model_object

    def predict(self, dataframe: DataFrame):
        logging.info("Entered the predict method of RentModel class")
        try:
            logging.info("Using the trained model to get predictions")
            transformed_features = self.preprocessing_object.transform(dataframe)
            preds_log = self.trainer_model_object.predict(transformed_features)
            # model was trained on log(rent) -> invert back to real rupees
            return np.expm1(preds_log)
        except Exception as e:
            raise RentException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trainer_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trainer_model_object).__name__}()"
