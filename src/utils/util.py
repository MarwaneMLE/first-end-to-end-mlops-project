import os
import sys
import pickle
import pandas as pd
import numpy as np 

from logger.logg import logging
from exceptions.exception import CustomException
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def save_object(file_path, object):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_object:
            pickle.dump(object, file_object) 

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(model.values()[i])

            model.fit(X_train, y_train)
        
            # Predict test data
            y_test_pred = model.predict(X_test)


            test_model_score = r2_score(y_pred=y_test_pred, y_true=y_test)

            report[list(models.keys())[i]] = test_model_score

    except Exception as e:
        logging.info("Exception occured during model training")
        raise CustomException(e, sys)












def load_object(file_path):
    try:
        with open(file_path, "rb") as file_object:
            return pickle.load(file_object) 

    except Exception as e:
        raise CustomException(e, sys)