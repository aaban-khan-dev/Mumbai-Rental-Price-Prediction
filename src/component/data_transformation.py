import sys
import os
import re
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import KNNImputer

from src.constant import *
from src.exception import RentException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join("artifact_folder", "data_transformation")
    transformed_train_file_path = os.path.join(data_transformation_dir, "train.npy")
    transformed_test_file_path = os.path.join(data_transformation_dir, "test.npy")
    transformed_object_file_path = os.path.join(data_transformation_dir, "preprocessing.pkl")


class DataTransformation:

    def __init__(self, valid_data_dir):
        self.valid_data_dir = valid_data_dir
        self.data_transformation_config = DataTransformationConfig()
        self.utils = MainUtils()

    @staticmethod
    def get_merged_batched_data(valid_data_dir: str) -> pd.DataFrame:
        """Read all valid CSV files and merge them into one dataframe."""
        try:
            raw_files = os.listdir(valid_data_dir)
            if not raw_files:
                raise RentException(f"No files found in the directory: {valid_data_dir}", sys)
            csv_data = []
            for filename in raw_files:
                data = pd.read_csv(os.path.join(valid_data_dir, filename))
                csv_data.append(data)
            return pd.concat(csv_data)
        except Exception as e:
            raise RentException(e, sys) from e

    def drop_schema_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Read schema.yaml and drop the columns listed under drop_columns."""
        try:
            _schema_config = self.utils.read_schema_config_file()
            return dataframe.drop(
                columns=_schema_config["drop_columns"], axis=1, errors="ignore"
            )
        except Exception as e:
            raise RentException(e, sys) from e

    def clean_and_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        All the cleaning/feature-engineering from the data_cleaning notebook:
        parse price/area/floor, engineer overlooking + nearby features, encode status,
        handle furnishing/facing nulls, dedup, and drop consumed raw columns.
        (Encoding + scaling are NOT here — they live in the preprocessor pipeline.)
        """
        try:
            logging.info("Cleaning and feature engineering started")

            # --- remove duplicate listings 
            before = len(df)
            df = df.drop_duplicates().reset_index(drop=True)
            logging.info(f"Dropped {before - len(df)} duplicate rows")

            # --- target: price_raw -> rent (numeric) ---
            def parse_price(x):
                if pd.isna(x):
                    return np.nan
                s = str(x).lower().replace("\u20b9", "").replace(",", "").strip()
                m = re.search(r"([\d.]+)", s)
                if not m:
                    return np.nan
                v = float(m[1])
                return (v * 1e7) if "cr" in s else ((v * 1e5) if ("lac" in s or "lakh" in s) else v)
            df["rent"] = df["price_raw"].apply(parse_price)
            df.drop(columns=["price_raw"], inplace=True, errors="ignore")

            #  bhk from title, then drop title 
            def parse_bhk(t):
                if pd.isna(t): return np.nan
                m = re.search(r"(\d+)\s*bhk", str(t).lower())
                return int(m[1]) if m else np.nan
            df["bhk"] = df["title"].apply(parse_bhk)
            df.drop(columns=["title"], inplace=True, errors="ignore")

            #  carpet area 
            def parse_area(x):
                if pd.isna(x): return np.nan
                s = str(x).replace(",", "")
                m = re.search(r"([\d]+)", s)
                return float(m[1]) if m else np.nan
            df["carpet_area"] = df["carpet_area_raw"].apply(parse_area)
            df.drop(columns=["carpet_area_raw"], inplace=True, errors="ignore")

            #  floor: floor_num, total_floors, floor_ratio 
            def parse_floor(x):
                if pd.isna(x): return (np.nan, np.nan)
                s = str(x).lower()
                total = np.nan
                mt = re.search(r"out of\s*(\d+)", s)
                if mt: total = int(mt[1])
                if "ground" in s: return (0, total)
                if "basement" in s: return (-1, total)
                mf = re.search(r"(\d+)", s)
                return (int(mf[1]) if mf else np.nan, total)
            df[["floor_num", "total_floors"]] = df["floor_raw"].apply(lambda x: pd.Series(parse_floor(x)))
            df["floor_ratio"] = np.where(
                (df["total_floors"].notna()) & (df["total_floors"] > 0),
                df["floor_num"] / df["total_floors"], np.nan
            )
            df.drop(columns=["floor_raw"], inplace=True, errors="ignore")

            #  property_type: merge house + villa into 'independent' 
            df["property_type"] = df["property_type"].replace({"house": "independent", "villa": "independent"})

            #  status_raw -> available_immediately
            df["available_immediately"] = (
                df["status_raw"].astype(str).str.strip().str.lower() == "immediately"
            ).astype(int)
            df.drop(columns=["status_raw"], inplace=True, errors="ignore")

            #  furnishing: normalise + mode-fill (encoding happens later) 
            df["furnishing"] = df["furnishing"].str.strip().str.title()
            df["furnishing"] = df["furnishing"].fillna(df["furnishing"].mode()[0])

            #  facing: null -> 'Unknown' 
            df["facing"] = df["facing"].fillna("Unknown").str.strip()

            #  overlooking -> 3 binary flags 
            ov = df["overlooking"].fillna("").str.lower()
            df["overlooks_garden"] = ov.str.contains("garden|park").astype(int)
            df["overlooks_pool"] = ov.str.contains("pool").astype(int)
            df["overlooks_main_road"] = ov.str.contains("main road").astype(int)
            df.drop(columns=["overlooking"], inplace=True, errors="ignore")

            #  nearby_raw -> metro_mins + proximity flags 
            def metro_mins(x):
                if pd.isna(x): return np.nan
                times = []
                for part in str(x).split("|"):
                    if "metro" in part.lower():
                        m = re.search(r"(\d+)\s*min", part.lower())
                        if m: times.append(int(m[1]))
                return min(times) if times else np.nan
            df["metro_mins"] = df["nearby_raw"].apply(metro_mins)
            df["has_metro"] = df["metro_mins"].notna().astype(int)
            n = df["nearby_raw"].fillna("").str.lower()
            df["near_school"] = n.str.contains("school").astype(int)
            df["near_hospital"] = n.str.contains("hospital|nursing").astype(int)
            df["near_mall"] = n.str.contains("mall").astype(int)
            df["near_bus"] = n.str.contains("bus").astype(int)
            df["near_railway"] = n.str.contains("railway").astype(int)
            df.drop(columns=["nearby_raw"], inplace=True, errors="ignore")

            #  drop rows with no rent or no area (can't model) 
            df = df[df["rent"].notna() & df["carpet_area"].notna()].copy()

            # --- IQR outlier trim on rent & area ---
            def iqr_keep(s, k=3):
                q1, q3 = s.quantile(.25), s.quantile(.75); iqr = q3 - q1
                return s.between(q1 - k * iqr, q3 + k * iqr)
            df = df[iqr_keep(df["rent"]) & iqr_keep(df["carpet_area"])].copy()

            # --- metro_mins: fill unreported with a 'far' value (has_metro flags it) ---
            far = df["metro_mins"].max()
            df["metro_mins"] = df["metro_mins"].fillna(far if pd.notna(far) else 60)

            logging.info(f"Cleaning complete. Shape: {df.shape}")
            return df

        except Exception as e:
            raise RentException(e, sys) from e

    #  build the preprocessor (encode + scale + impute) 
    def get_preprocessor(self) -> ColumnTransformer:
        """
        ColumnTransformer that:
          - one-hot encodes property_type, locality, facing
          - ordinal encodes furnishing (Unfurnished<Semi<Furnished)
          - KNN-imputes + scales the continuous numerics
          - passes the binary flags through unchanged
        Saved as one object and reused at prediction time.
        """
        try:
            onehot_cols = ["property_type", "locality", "facing"]
            ordinal_cols = ["furnishing"]
            numeric_cols = ["bhk", "carpet_area", "floor_num", "total_floors", "floor_ratio", "metro_mins"]
            binary_cols = ["available_immediately", "has_metro",
                           "overlooks_garden", "overlooks_pool", "overlooks_main_road",
                           "near_school", "near_hospital", "near_mall", "near_bus", "near_railway"]

            furnish_order = [["Unfurnished", "Semi-Furnished", "Furnished"]]

            # numeric branch: KNN impute (for area/floor nulls) then scale
            numeric_pipe = Pipeline([
                ("imputer", KNNImputer(n_neighbors=5)),
                ("scaler", StandardScaler()),
            ])

            return ColumnTransformer(transformers=[
                ("oh", OneHotEncoder(handle_unknown="ignore", drop="first"), onehot_cols),
                ("ord", OrdinalEncoder(categories=furnish_order), ordinal_cols),
                ("num", numeric_pipe, numeric_cols),
                ("bin", "passthrough", binary_cols),
            ])
        except Exception as e:
            raise RentException(e, sys) from e

    def initiate_data_transformation(self):
        """Full transformation: read -> drop -> clean -> split -> fit preprocessor -> arrays."""
        logging.info("Initiated data transformation")
        try:
            dataframe = self.get_merged_batched_data(self.valid_data_dir)
            dataframe = self.drop_schema_columns(dataframe)
            dataframe = self.clean_and_engineer(dataframe)

            # target log-transform (rent is right-skewed)
            X = dataframe.drop(columns=[TARGET_COLUMN], axis=1)
            y = np.log1p(dataframe[TARGET_COLUMN])   # model log(rent)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            preprocessor = self.get_preprocessor()
            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            preprocessor_path = self.data_transformation_config.transformed_object_file_path
            os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
            self.utils.save_object(preprocessor_path, obj=preprocessor)

            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr = np.c_[X_test_arr, np.array(y_test)]

            logging.info("Data transformation completed")
            return (train_arr, test_arr, preprocessor_path)

        except Exception as e:
            raise RentException(e, sys) from e
