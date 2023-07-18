import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

class DataTranformation:

    def initiate_data_tranformation(self,day_train_data_path,day_test_data_path,hour_train_data_path,hour_test_data_path):
        try:
            day_train_df = pd.read_csv(day_train_data_path)
            day_test_df = pd.read_csv(day_test_data_path)
            hour_train_df = pd.read_csv(hour_train_data_path)
            hour_test_df = pd.read_csv(hour_test_data_path)

            logging.info("Read train and test data completed")

            return(
                day_train_df,
                day_test_df,
                hour_train_df,
                hour_test_df
            )

        except Exception as e:
            raise CustomException(e,sys)