import os
import sys
import pickle
import dill
import datetime
import math

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





# -------- ZONE --------
def get_zone(cy, line1, line2):
    if cy < line1:
        return "Z1"
    elif cy < line2:
        return "Z2"
    else:
        return "Z3"


# -------- TIME --------
def get_time_of_day():
    hour = datetime.datetime.now().hour
    return "Day" if 6 <= hour <= 18 else "Night"


# -------- SPEED --------
def calculate_speed(track_id, cx, cy, y1, y2, mapped_label, object_tracks, time_diff):

    if mapped_label == "Human":
        real_height = 1.7
    elif mapped_label == "Vehicle":
        real_height = 1.5
    else:
        real_height = 1.0

    height_pixels = y2 - y1

    if height_pixels > 0:
        meters_per_pixel = real_height / height_pixels

        if track_id in object_tracks:
            prev_cx, prev_cy = object_tracks[track_id]

            pixel_distance = math.sqrt(
                (cx - prev_cx)**2 + (cy - prev_cy)**2
            )

            distance_meters = pixel_distance * meters_per_pixel
            speed_mps = distance_meters / time_diff
            speed = speed_mps * 3.6

            speed = min(speed, 120)
        else:
            speed = 0

        object_tracks[track_id] = (cx, cy)

    else:
        speed = 0

    return round(speed, 2)

