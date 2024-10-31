import pandas as pd
import numpy as np 
from logger.logg import logging
from exceptions.exception import CustomException

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from utils.util import save_object


@dataclass
class DataTransfomationConfig:
    preprocessor_object_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransfomation:
    def __init__(self):
        self.data_transformation_config = DataTransfomationConfig()


    def get_data_transformation(self):
        try:
            logging.info("Data transformation start")

            # Separete categorical and numerical columns
            categorical_cols = ["cut", "color", "clarity"]
            numerical_cols = ["carat", "depth", "table", "x", "y", "z"]

            # Custum ranking for each ordinal feature
            cut_categories = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
            color_categories = ["D", "E", "F", "G", "H", "I", "J"]
            clarity_categories =  ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

            logging.info("Pipeline initiated")
            

            ## Numerical pipeline
            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]  
            )
            
            ## Categorical pipeline
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ordinalencoder", OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
                    ("scaler", StandardScaler())
                ]
            )
 

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_cols),
                ("cat_pipeline", cat_pipeline, categorical_cols)
            ])

            return preprocessor
        
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")
            raise CustomException(e, sys)
        
    
    def initialize_data_transformation(self, train_path, test_path):
        try:
            # This data come from data ingestion from artifact dir
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data")
            logging.info(f"Train DataFrame head : \n {train_df.head().to_string()}")
            logging.info(f"Test DataFrame head : \n {test_df.head().to_string()}")

            preprocessing_object = self.get_data_transformation()

            target_column_name = "price"
            drop_columns = [target_column_name, "id"]
             
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            input_feature_train_arr = preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_object.transform(input_feature_test_df)

            #logging.info("Preprocesing in applyed on train and test data")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_object_file_path,
                object=preprocessing_object
            )
            logging.info("The preprocessing pickle file is saved")

            return (
                train_arr,
                test_arr
            )
        
        except Exception as e:
            logging.info("Exception occured  in the initiate_dataframe")
            raise CustomException(e, sys)