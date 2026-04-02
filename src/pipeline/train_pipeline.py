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
        