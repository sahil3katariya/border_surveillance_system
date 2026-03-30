import os
import sys
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj

from src.components.data_ingestion import DataIngestion

from sklearn.compose import ColumnTransformer,make_column_transformer
from sklearn.pipeline import Pipeline , make_pipeline

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
from sklearn.preprocessing import MinMaxScaler,StandardScaler

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_file_obj_path = os.path.join('artifacts','preprocessor.pkl')
    label_encoder_file_obj_path = os.path.join('artifacts','label_encoder.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            df = pd.read_csv('notebook/border_surveillance_dataset.csv')
            
            num_cols = [c for c in df.columns if df[c].dtype != 'object']
            
            cat_cols = [c for c in df.columns if df[c].dtype == 'object']

            if 'risk' in cat_cols:
                cat_cols.remove('risk')
            
            
            preprocessor = ColumnTransformer([
                ('cat_transform',OneHotEncoder(drop='first',sparse_output=False),cat_cols),
                ('scaler',StandardScaler(),num_cols)
            ],remainder='passthrough')

            Label_encoder = LabelEncoder()
            
            return preprocessor, Label_encoder
            
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_set = pd.read_csv(train_path)
           
            test_set = pd.read_csv(test_path)
           

            train_set_input_features = train_set.drop(columns=['risk'])
        
            train_set_output_feature = train_set['risk']
            
            test_set_input_features = test_set.drop(columns=['risk'])
            test_set_output_feature = test_set['risk']

            preprocessor_obj,label_encoder_obj = self.get_data_transformer_object()

            train_input_arr = preprocessor_obj.fit_transform(train_set_input_features)
            test_input_arr = preprocessor_obj.transform(test_set_input_features)

            train_label_encoder_arr = label_encoder_obj.fit_transform(train_set_output_feature)
            test_label_encoder_arr = label_encoder_obj.transform(test_set_output_feature)

            train_arr = np.c_[train_input_arr,train_label_encoder_arr]
            test_arr = np.c_[test_input_arr,test_label_encoder_arr]

            save_obj(self.data_transformation_config.preprocessor_file_obj_path,preprocessor_obj)

            save_obj(self.data_transformation_config.label_encoder_file_obj_path,label_encoder_obj)

            return (
                train_arr,test_arr,self.data_transformation_config.preprocessor_file_obj_path
            )
    
        except Exception as e:
            raise CustomException(e,sys)
        


