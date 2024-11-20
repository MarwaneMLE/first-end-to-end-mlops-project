import os
import sys
import pandas as pd
import numpy as np 
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import pickle
 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

from logger.logg import logging
from exceptions.exception import CustomException
from utils.util import load_object


class ModelEvaluation:
    def __init__(self):
        logging.info("Evaluation started")
    
    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_absolute_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        logging.info("Evaluation metrics  capture")
        return rmse, mae, r2
     
    """
    def initiate_model_evaluation(self, train_arr, test_arr):
        try:
            X_test, y_test = (test_arr[:,:-1], test_arr[:,-1])

            model_path = os.path.join("artifacts","model.pkl")
            model = load_object(model_path)
            
            # mlflow.set_registry_uri()

            logging.info("Model has regester")

            tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

            print(tracking_url_type_store)

            with mlflow.start_run():
                prediction = model.predict(X_test)

                rmse, mae, r2 = self.eval_metrics(y_test, prediction)

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)

                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
                else:
                    mlflow.sklearn.log_model(model, "model")

        except Exception as e:
            # logging.info("Exception")
            raise CustomException(e, sys)
    """
    def initiate_model_evaluation(self,train_array,test_array):

        try:
            X_test,y_test=(test_array[:,:-1], test_array[:,-1])

            model_path=os.path.join("artifacts","model.pkl")
            model=load_object(model_path)

            #mlflow.set_registry_uri("")
            
            logging.info("model has register")

            tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme

            print(tracking_url_type_store)



            with mlflow.start_run():

                prediction=model.predict(X_test)

                (rmse,mae,r2)=self.eval_metrics(y_test,prediction)

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)

                # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
                else:
                    mlflow.sklearn.log_model(model, "model")


        except Exception as e:
            raise CustomException(e,sys)
      