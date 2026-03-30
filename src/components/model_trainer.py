import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_ingestion import DataIngestion

# Basic ML models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier


# Boosting libraries
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Evaluation & tuning
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix,classification_report
from sklearn.model_selection import GridSearchCV

from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate_model,save_obj


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            x_train,x_test,y_train,y_test = (
                train_arr[:,:-1],
                test_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,-1]
            )

            models = {

                "Logistic Regression": {
                    "model": LogisticRegression(max_iter=1000),
                    "params": {
                        "C": [0.01, 0.1, 1, 10],
                        "penalty": ["l2"]
                    }
                },

                "Naive Bayes": {
                    "model": GaussianNB(),
                    "params": {}
                },
                
                "KNN": {
                    "model": KNeighborsClassifier(),
                    "params": {
                        "n_neighbors": [3, 5, 7],
                        "weights": ["uniform", "distance"],
                        "metric": ["minkowski"]
                    }
                },
                
                "Decision Tree": {
                    "model": DecisionTreeClassifier(),
                    "params": {
                        "criterion": ["gini", "entropy"],
                        "max_depth": [None, 10, 20, 30],
                        "min_samples_split": [2, 5, 10]
                    }
                },

                "Random Forest": {
                    "model": RandomForestClassifier(),
                    "params": {
                        "n_estimators": [100, 200],
                        "max_depth": [None, 10, 20],
                        "min_samples_split": [2, 5],
                        "class_weight": ["balanced"]
                    }
                },

                "AdaBoost": {
                    "model": AdaBoostClassifier(),
                    "params": {
                        "n_estimators": [50, 100, 200],
                        "learning_rate": [0.01, 0.1, 1]
                    }
                },
                
                "Gradient Boosting": {
                    "model": GradientBoostingClassifier(),
                    "params": {
                        "n_estimators": [100, 200],
                        "learning_rate": [0.05, 0.1],
                        "max_depth": [3, 5]
                    }
                },
                
                "XGBoost": {
                    "model": XGBClassifier(eval_metric='mlogloss'),
                    "params": {
                        "n_estimators": [100, 200],
                        "learning_rate": [0.05, 0.1],
                        "max_depth": [3, 6],
                        "subsample": [0.8, 1.0]
                    }
                },

                "CatBoost": {
                    "model": CatBoostClassifier(verbose=0, allow_writing_files=False),
                    "params": {
                        "iterations": [100, 200],
                        "learning_rate": [0.05, 0.1],
                        "depth": [4, 6]
                    }
                }
            }

            model_report,best_models= evaluate_model(x_train,y_train,x_test,y_test,models)

            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = best_models[best_model_name]

            if best_model_score < 0 : 
                raise CustomException('no best model found',sys)
            
            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            y_pred = best_model.predict(x_test)

            # acc = accuracy_score(y_test,predicted)
            # return   acc  

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            print("Accuracy:", accuracy)
            print("Precision:", precision)
            print("Recall:", recall)
            print("F1 Score:", f1)

            # Confusion Matrix
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
            
            # Full Report
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))

        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == '__main__':
    data_ingestion_obj = DataIngestion()
    train_data , test_data = data_ingestion_obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr , test_arr , _ = data_transformation.initiate_data_transformation(train_data,test_data)

    model_trainer = ModelTrainer()
    acc = model_trainer.initiate_model_trainer(train_arr,test_arr)

    print(acc)


