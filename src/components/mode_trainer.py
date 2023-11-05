import os
import sys

import numpy as np
import pandas as pd
from sklearn.svm import SVC

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model
from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    training_model_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting dependent and independent data from train and test data")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            model = SVC(kernel="rbf")
            model_report = evaluate_model(x_train, y_train, x_test, y_test)

            logging.info(f"Model Report : {model_report}")
            print(f"Model Report : {model_report}")

            save_object(file_path=self.model_trainer_config.training_model_path, obj=model)
        except Exception as e:
            logging.info("An error has occurred at Model Training")
            raise CustomException(e, sys)
