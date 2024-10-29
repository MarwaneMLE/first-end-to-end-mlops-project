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
"""from logger import logg

@dataclass
class ModelEvaluationConfig:
    pass


class ModelEvaluation:
    def __init__(self):
        pass

    def initiate_model_evaluation(self):
        try:
            pass
        except Exception as e:
            logging.info()
            raise CustomException(e, sys)"""