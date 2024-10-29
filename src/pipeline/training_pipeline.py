import os
import sys
#from logger.logg import logging
#from exceptions.exception import customexception
#import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransfomation
from src.components.model_training import ModelTrainer 

object = DataIngestion()

train_path, test_path = object.initiate_data_ingestion()

data_transformation = DataTransfomation()
 
train_arr, test_arr = data_transformation.initialize_data_transformation(train_path, test_path)

model_trainer_object = ModelTrainer()
model_trainer_object.initiate_model_trainig(train_arr, test_arr)