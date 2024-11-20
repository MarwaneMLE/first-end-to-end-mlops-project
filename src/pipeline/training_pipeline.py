import os
import sys
#from logger.logg import logging
#from exceptions.exception import customexception
#import pandas as pd

from components.data_ingestion import DataIngestion
from components.data_transformation import DataTransfomation
from components.model_training import ModelTrainer 
from components.model_evaluation import ModelEvaluation


object = DataIngestion()

train_path, test_path = object.initiate_data_ingestion()

data_transformation = DataTransfomation()
 
train_arr, test_arr = data_transformation.initialize_data_transformation(train_path, test_path)

model_trainer_object = ModelTrainer()
model_trainer_object.initiate_model_trainig(train_arr, test_arr)

# Evaluation the model
model_trainer_object = ModelTrainer()
model_trainer_object.initiate_model_trainig(train_arr, test_arr)

model_eval_object = ModelEvaluation()
model_eval_object.initiate_model_evaluation(train_arr, test_arr)