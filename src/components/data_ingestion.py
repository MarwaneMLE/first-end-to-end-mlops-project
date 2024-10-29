import os
import sys
import pandas as pd
import numpy as np 
from dataclasses import dataclass
from pathlib import Path

from logger.logging import logging
from exceptions.exception import CustomException 
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")

        try:
            logging.info("data ingestion started")
            #data = pd.read_csv("/home/marwane/mlops-projects/first-end-to-end-mlops-project/data/playground-series-s3e8/train.csv")
            data = pd.read_csv("https://raw.githubusercontent.com/sunnysavita10/fsdsmendtoend/main/notebooks/data/gemstone.csv")
            logging.info("Reading my dataframe")

            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path))),
            data.to_csv(self.ingestion_config.raw_data_path, index=False),
            logging.info("I have saved the raw dataset in artifact folder")

            logging.info("Here I Have performed train test split")         
            
            train_data, test_data = train_test_split(data, test_size=0.25)
            logging.info("Train test split complated") 

            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Data Ingestion is complated") 

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info("Exception Occured")
            raise CustomException(e, sys)

"""
if __name__ == "__main__":
    object = DataIngestion()  
    object.initiate_data_ingestion()"""