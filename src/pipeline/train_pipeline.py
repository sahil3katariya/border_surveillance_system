import os
import sys

from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        self.ingestion_obj = DataIngestion()
        self.transformation_obj = DataTransformation()
        self.model_trainer_obj = ModelTrainer()

    def train_pipeline(self):
        try:
            train_data_path,test_data_path = self.ingestion_obj.initiate_data_ingestion()
            train_arr,test_arr,preprocessor_file_path,label_encoder_file_path = self.transformation_obj.initiate_data_transformation(train_data_path,test_data_path)
            self.model_trainer_obj.initiate_model_trainer(train_arr,test_arr)


        except Exception as e:
            raise CustomException(e,sys)