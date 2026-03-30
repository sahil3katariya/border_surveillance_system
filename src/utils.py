import os
import sys
import pickle
import dill

from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


def save_obj(file_path,obj):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        
        with open(file_path,'wb') as f:
            pickle.dump(obj,f)

            
    except Exception as e:
        raise CustomException(e,sys)


def  evaluate_model(x_train,y_train,x_test,y_test,models):
    try:
        report = {}
        best_models = {}

        for name,data in models.items():
            model = data['model']
            params = data['params']

            if params:
                grid = GridSearchCV(
                    model,
                    params,
                    cv = 6,
                    scoring='accuracy',
                    n_jobs=-1
                )
                grid.fit(x_train,y_train)
                best_model = grid.best_estimator_

            else:
                model.fit(x_train,y_train)
                best_model = model


            y_train_pred = best_model.predict(x_train)
            y_test_pred = best_model.predict(x_test)

            train_accuracy_score = accuracy_score(y_train,y_train_pred)
            test_accuracy_score = accuracy_score(y_test,y_test_pred)

            report[name] = test_accuracy_score
            best_models[name] = best_model

        return report,best_models

    except Exception as e:
        raise CustomException(e,sys)


