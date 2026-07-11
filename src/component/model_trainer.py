import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

from src.constant import *
from src.exception import RentException
from src.logger import logging
from src.utils.main_utils import MainUtils
from src.model.estimator import RentModel   # changed: RentModel now lives in model/estimator.py

from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(artifact_folder, "model_trainer")
    trained_model_path = os.path.join(model_trainer_dir, "trained_model", "model.pkl")
    expected_accuracy = 0.6
    model_config_file_path = os.path.join("config", "model.yaml")



class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.utils = MainUtils()

    def evaluate_models(self, X_train, y_train, X_test, y_test, models):
        try:
            model_report = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_test_pred = model.predict(X_test)
                # models are trained on log(rent); invert with expm1 so R² is on the
                # real rupee scale 
                test_model_score = r2_score(np.expm1(y_test), np.expm1(y_test_pred))
                model_report[name] = test_model_score
            return model_report
        except Exception as e:
            raise RentException(e, sys) from e

    def cross_validate_model(self, model, X, y):
        """5-fold CV to confirm the score is stable and not a fluke of the train/test split.
        """
        try:
            from sklearn.metrics import make_scorer

            # custom scorer: invert log before computing R², so CV reports rupee-scale R²
            def rupee_r2(y_true_log, y_pred_log):
                return r2_score(np.expm1(y_true_log), np.expm1(y_pred_log))

            scorer = make_scorer(rupee_r2)
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=kf, scoring=scorer, n_jobs=-1)
            logging.info(f"Cross-val R² (rupee scale): mean={scores.mean():.3f}, std={scores.std():.3f}")
            return scores.mean(), scores.std()
        except Exception as e:
            raise RentException(e, sys) from e

    def fine_tune_model(self, best_model_object, best_model_name, X_train, y_train):
        try:
            model_param_grid = self.utils.read_yaml_file(
                self.model_trainer_config.model_config_file_path
            )["model_selection"]["model"][best_model_name]["search_param_grid"]
            grid_search = GridSearchCV(best_model_object, param_grid=model_param_grid,
                                       cv=5, n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            logging.info(f"Best params for {best_model_name}: {best_params}")
            return best_model_object.set_params(**best_params)
        except Exception as e:
            raise RentException(e, sys) from e

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # rental regression models (matches model.yaml keys)
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(),
                "Random Forest Regressor": RandomForestRegressor(random_state=42, n_jobs=-1),
                "XGBoost Regressor": xgb.XGBRegressor(random_state=42, n_jobs=-1),
                "LightGBM Regressor": lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
            }

            logging.info(f"Loading preprocessor from: {preprocessor_path}")
            preprocessor = self.utils.load_object(file_path=preprocessor_path)

            # compare all models on the test split
            model_report = self.evaluate_models(X_train, y_train, X_test, y_test, models=models)
            logging.info(f"Model report (test R²): {model_report}")

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            logging.info(f"Best model: {best_model_name} (test R²={best_model_score:.3f})")

            # cross-validation on the best model — confirms the score is real
            cv_mean, cv_std = self.cross_validate_model(best_model, X_train, y_train)
            logging.info(f"{best_model_name} CV R² = {cv_mean:.3f} ± {cv_std:.3f}")

            # hyperparameter tuning on the best model
            best_model = self.fine_tune_model(
                best_model_object=best_model,
                best_model_name=best_model_name,
                X_train=X_train,
                y_train=y_train,
            )

            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            # invert log -> rupee scale for the final reported R² (matches notebook)
            final_score = r2_score(np.expm1(y_test), np.expm1(y_pred))
            logging.info(f"Final tuned {best_model_name} test R² (rupee scale) = {final_score:.3f}")

            if final_score < self.model_trainer_config.expected_accuracy:
                raise RentException("No best model found with score greater than expected accuracy", sys)

            custom_model = RentModel(preprocessing_object=preprocessor, trainer_model_object=best_model)

            logging.info(f"Saving best model to: {self.model_trainer_config.trained_model_path}")
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_path), exist_ok=True)
            self.utils.save_object(
                file_path=self.model_trainer_config.trained_model_path, obj=custom_model
            )

            # changed: removed the redundant S3 folder-sync here.
            # The trained model is now pushed to S3 by the training pipeline via
            # RentEstimator (start_model_pusher), so we only save locally here and
            # return the path.
            return final_score, self.model_trainer_config.trained_model_path

        except Exception as e:
            raise RentException(e, sys)