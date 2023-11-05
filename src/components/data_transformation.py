import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join("Artifacts","Preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):

        try:
            logging.info("Data Transformation Initiated")

            categorical_columns = ['City', 'AQI_Bucket']
            numerical_columns = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2',
                                    'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']

            # numerical pipeline

            num_pipeline = Pipeline(
                steps=[
                    ("Scaler", StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("OrdinalEncoder", OrdinalEncoder()),
                    ("Scaler", StandardScaler())

                ]
                )

            preprocessor = ColumnTransformer(
                    [
                        ("numerical pipeline", num_pipeline, numerical_columns),
                        ("Categorical Pipeline", cat_pipeline, categorical_columns)
                    ]
                )
            return preprocessor
            logging.info("Pipeline completed")

        except Exception as e:
            logging.info("Error  in data Transformation")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Reading Train and Test data has completed")
            logging.info(f"Train Dataframe Overview:\n {train_data.head().to_String()}")
            logging.info(f"Test Dataframe Overview:\n {test_data.head().to_string()}")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformation_object()

            target_column = "AQI_Bucket"
            drop_columns = [target_column, "Date"]

            input_feature_train_data = train_data.drop(columns=drop_columns, axis=1)
            target_feature_train_data = train_data[target_column]

            input_feature_test_data = train_data.drop(columns=drop_columns, axis=1)
            target_feature_test_data = test_data[target_column]

            logging.info("Applying Preprocessing object on training and testing datasets")

            # transforming using preprocessing objects

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_data)
            input_feature_test_data = preprocessing_obj.fit_transform(input_feature_test_data)

            logging.info("Applying preprocessing objects on training and testing datasets")

            train_arr = np.c_[input_feature_train_arr, np.arr(target_feature_train_data)]
            test_arr = np.c_[input_feature_test_data, np.arr(target_feature_test_data)]

            save_object(file_path=self.data_transformation_config.preprocessor_obj_path, obj=preprocessing_obj)
            logging.info("Preprocessor File is saved")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_path
            )
        except Exception as e:
            logging.info("An error occurred in Initiate Data Transformation ")
            raise CustomException(e, sys)
