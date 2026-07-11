import sys
import numpy as np
import pandas as pd

import shap

from src.exception import RentException
from src.logger import logging


class ShapExplainer:
    """
    Computes a per-prediction SHAP breakdown ("why this price") and maps the
    one-hot / transformed columns back to clean, human-readable feature names.

    The saved model is a RentModel wrapper (preprocessor + tree model).
    SHAP runs on the RAW tree model + the TRANSFORMED features, so this class
    reaches into the wrapper for both pieces.
    """

    # map the transformed column prefixes/names to friendly labels shown on the UI
    FRIENDLY_NAMES = {
        "property_type": "Property type",
        "locality": "Locality",
        "facing": "Facing direction",
        "furnishing": "Furnishing",
        "bhk": "Number of bedrooms (BHK)",
        "carpet_area": "Carpet area",
        "floor_num": "Floor number",
        "total_floors": "Building height",
        "floor_ratio": "Floor position",
        "metro_mins": "Distance to metro",
        "has_metro": "Metro nearby",
        "available_immediately": "Immediately available",
        "overlooks_garden": "Garden view",
        "overlooks_pool": "Pool view",
        "overlooks_main_road": "Main-road view",
        "near_school": "Near a school",
        "near_hospital": "Near a hospital",
        "near_mall": "Near a mall",
        "near_bus": "Near a bus stop",
        "near_railway": "Near a railway station",
    }

    def __init__(self, rent_model):
        """
        rent_model: the loaded RentModel 
        """
        try:
            self.preprocessor = rent_model.preprocessing_object
            self.model = rent_model.trainer_model_object
            # TreeExplainer is exact + fast for XGBoost/RF/LightGBM
            self.explainer = shap.TreeExplainer(self.model)
        except Exception as e:
            raise RentException(e, sys)

    def _base_feature(self, transformed_name: str) -> str:
        """
        Turn a transformed column name into its base feature.
        """
        # strip the transformer prefix (oh__ / num__ / ord__ / bin__)
        name = transformed_name.split("__", 1)[-1]

        # for one-hot columns like 'locality_bandra-west', keep the part before the
        # first underscore that starts the category value. We match against known bases.
        return next((base for base in self.FRIENDLY_NAMES
                     if name == base or name.startswith(base + "_")), name)

    def explain(self, input_df: pd.DataFrame, top_n: int = 8) -> dict:
        """
        Compute the SHAP breakdown for a single prepared input row.

        Returns a dict:
        {
          "base_value": <rupee baseline>,
          "contributions": [ {"feature": "Locality", "impact": +8200.0}, ... ]  # sorted by |impact|
        }
        Impacts are in RUPEES (SHAP is computed in log space, then converted).
        """
        try:
            # 1. transform the input the same way the model was trained
            transformed = self.preprocessor.transform(input_df)
            feature_names = self.preprocessor.get_feature_names_out()

            # 2. SHAP values (in LOG space, since the model predicts log(rent))
            shap_values = self.explainer.shap_values(transformed)
            shap_row = np.array(shap_values)[0]                # one row
            base_log = float(self.explainer.expected_value)    # log-space baseline
            pred_log = base_log + shap_row.sum()

            # 3. group transformed-column contributions back to base features (LOG space)
            grouped_log = {}
            for name, val in zip(feature_names, shap_row):
                base = self._base_feature(name)
                grouped_log[base] = grouped_log.get(base, 0.0) + float(val)

            # 4. convert each feature's LOG contribution to an approximate RUPEE impact.
            #    rent = expm1(log_pred). We attribute the rupee change of each feature as
            #    the difference it makes to the final rupee prediction.
            base_rupee = np.expm1(base_log)
            pred_rupee = np.expm1(pred_log)
            total_log_effect = pred_log - base_log

            contributions = []
            for base, log_val in grouped_log.items():
                # proportional split of the total rupee change by each feature's log share
                if total_log_effect != 0:
                    rupee_impact = (log_val / total_log_effect) * (pred_rupee - base_rupee)
                else:
                    rupee_impact = 0.0
                contributions.append({
                    "feature": self.FRIENDLY_NAMES.get(base, base),
                    "impact": round(float(rupee_impact), 0),
                })

            # 5. sort by absolute impact, keep the top N most influential
            contributions.sort(key=lambda d: abs(d["impact"]), reverse=True)
            contributions = contributions[:top_n]

            logging.info(f"SHAP breakdown computed for prediction ({len(contributions)} factors)")

            return {
                "base_value": round(float(base_rupee), 0),
                "predicted_value": round(float(pred_rupee), 0),
                "contributions": contributions,
            }

        except Exception as e:
            raise RentException(e, sys)
