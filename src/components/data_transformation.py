import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    PreproscessorPath = os.path.join('artifacts', 'preprocessor.pkl')


class Preprocessor:
    def __init__(self):
        self.Preprocessor_config = DataTransformationConfig()

    def PreprocessorSetup(self):
        """
        This function sets up preprocessing pipelines.
        """
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def ApplyPreprocessing(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data successfully")

            preprocessor_obj = self.PreprocessorSetup()

            target_column = "math_score"
            input_train_df = train_df.drop(columns=[target_column])
            target_train_df = train_df[target_column]

            input_test_df = test_df.drop(columns=[target_column])
            target_test_df = test_df[target_column]

            logging.info("Fitting and transforming training and test data.")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_test_df)

            train_array = np.c_[input_feature_train_arr, np.array(target_train_df)]
            test_array = np.c_[input_feature_test_arr, np.array(target_test_df)]

            save_object(
                file_path=self.Preprocessor_config.PreproscessorPath,
                obj=preprocessor_obj
            )

            logging.info(f"Preprocessor saved at {self.Preprocessor_config.PreproscessorPath}")

            return train_array, test_array, self.Preprocessor_config.PreproscessorPath

        except Exception as e:
            raise CustomException(e, sys)

    