import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join("artifacts", "train.csv")
    test_data_path = os.path.join("artifacts", "test.csv")
    raw_data_path = os.path.join("artifacts", "raw.csv")


# create a class for data ingestion
class DataIngestion:
    def __init__(self):
        self.ingestionConfig = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion method starts")
        try:
            data = pd.read_csv(os.path.join("notebooks/data", "Air_quality_data,csv"))
            logging.info("Dataset read as pandas Dataframe")

            os.makedirs(
                os.path.dirname(self.ingestionConfig.raw_data_path),
                exist_ok=True
            )
            data.to_csv(
                self.ingestionConfig.raw_data_path,
                index=False
            )
            logging.info("train Test Split")
            train_data, test_data = train_test_split(data, test_size=0.3)

            train_data.to_csv(
                self.ingestionConfig.train_data_path,
                index=False,
                header=True
            )

            test_data.to_csv(
                self.ingestionConfig.test_data_path,
                index=False,
                header=True
            )

            logging.info("Data Ingestion is completed")

            return (
                self.ingestionConfig.train_data_path,
                self.ingestionConfig.test_data_path
                    )

        except Exception as e:
            logging.info("Exception occurred at Data Ingestion stage")
            raise CustomException(e, sys)
