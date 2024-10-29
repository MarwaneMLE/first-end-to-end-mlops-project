import os
import sys 
import pandas as pd
import numpy as np 
from logger.logg import logging
from exceptions.exception import CustomException

from dataclasses import dataclass       
from pathlib import Path

from utils.util import save_object, evaluate_model
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainer()

    def initiate_model_trainig(self, train_arr, test_arr):
        try:
            logging.info("Spliting dependent in independent features")
            X_train , y_train, X_test, y_test = (*
                train_arr[:,:-1], 
                train_arr[:,-1],                   
                test_arr[:,:-1], 
                test_arr[:,-1],                   
            )

            models = {
                "LinearRegression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "ElasticNet": ElasticNet(),
                #"RandomForest": RandomForestRegressor(),
                #"XGBoost": XGBRegressor()
            }

            model_report: dict = evaluate_model(X_train , y_train, X_test, y_test)
            print(model_report)
            print("\n==========================================")

            # Get the best model score from dictionary
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            
            print(f'Best Model Found Is : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found Is : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
        except Exception as e:
            logging.info()
            raise CustomException(e, sys)