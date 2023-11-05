import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scale = preprocessor.transform(features)
            prediction = model.predict(data_scale)

            return prediction
        except Exception as e:
            logging.info("Error Occurred at prediction pipeline")
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                City:object,
                Date:object,
                PM2DOT5:float,
                PM10:float,
                NO:float,
                NO2:float,
                NOx:float,
                NH3:float,
                CO:float,
                SO2:float,
                O3:float,
                Benzene:float,
                Toluene:float,
                Xylene:float,
                 ):
        self.City = City
        self.Date = Date
        self.PM2DOT5 = PM2DOT5
        self.PM10 = PM10
        self.NO = NO
        self.NO2 = NO2
        self.NOx = NOx
        self.NH3 = NH3
        self.CO = CO
        self.SO2 = SO2
        self.O3 = O3
        self.Benzene = Benzene
        self.Toluene = Toluene
        self.Xylene = Xylene

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "City": [self.City],
                "Date": [self.Date],
                "PM2.5": [self.PM2DOT5],
                "PM10": [self.PM10],
                "NO": [self.NO],
                "NO2": [self.NO2],
                "NOx": [self.NOx],
                "NH3": [self.NH3],
                "CO": [self.CO],
                "SO2": [self.SO2],
                "O3": [self.O3],
                "Benzene": [self.Benzene],
                "Toluene": [self.Toluene],
                "Xylene": [self.Xylene],
            }

            dataframe = pd.DataFrame(custom_data_input_dict)
            logging.info("Dataframe Gathered")
            return dataframe
        except Exception as e:
            logging.info("Exception Occurred at prediction pipeline")
            raise CustomException(e, sys)

