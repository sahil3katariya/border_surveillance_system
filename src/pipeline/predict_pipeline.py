import os
import sys

from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np

import joblib


class Predict_pipeline:
    def __init__(self):
        try:
            self.label_encoder = joblib.load('artifacts/label_encoder.pkl')
            self.ml_model = joblib.load('artifacts/model.pkl')
            self.preprocessor = joblib.load('artifacts/preprocessor.pkl')

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, data: dict):
        try:
            df = pd.DataFrame([data])

            

            preprocessor_out = self.preprocessor.transform(df)
            
            ml_model_out = self.ml_model.predict(preprocessor_out)
            ml_model_out =  ml_model_out.astype(int)
            label_encoder_out = self.label_encoder.inverse_transform(ml_model_out)

            return label_encoder_out[0]

        except Exception as e:
            raise CustomException(e, sys)
