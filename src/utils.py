import os
import sys
import pickle
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(x_train, y_train, x_test, y_test, model):
    try:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        # get the accuracy score from the model
        model_score = accuracy_score(y_test, y_pred)
        report = model_score
        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info("Exception occurred in load_object function calls")
        raise CustomException(e, sys)
