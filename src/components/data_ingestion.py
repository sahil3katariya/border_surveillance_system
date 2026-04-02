import sys
import os
import pandas as pd
from src.logger import logging
from src.exception import CustomException

from sklearn.model_selection import train_test_split    
from dataclasses import dataclass


@dataclass 
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts','training_data.csv')
    test_data_path = os.path.join('artifacts','test_data.csv')
    raw_data_path = os.path.join('artifacts','raw_data.csv')
    artifact_path = os.path.join('artifacts')

class DataIngestion:
    def __init__(self):
        self.IngestionConfig = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        try:
            logging.info('Data ingestion started')
            df = pd.read_csv('notebook/border_dataset_realistic_1500.csv')
            os.makedirs(self.IngestionConfig.artifact_path,exist_ok=True)

            logging.info('Read Dataset As DataFrame')
            
            df.to_csv(self.IngestionConfig.raw_data_path,index=False)
            
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=43)
            
            logging.info('Data Divided Successfully')

            train_set.to_csv(self.IngestionConfig.train_data_path,index=False)
            test_set.to_csv(self.IngestionConfig.test_data_path,index=False)

            logging.info('Data Ingestion Completed')
            
            return (
                self.IngestionConfig.train_data_path,
                self.IngestionConfig.test_data_path
            )
        

        except Exception as e:
            raise CustomException(e,sys)




