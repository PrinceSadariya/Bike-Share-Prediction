import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    day_train_data_path:str = os.path.join("artifacts","day_train.csv")
    day_test_data_path:str = os.path.join("artifacts","day_test.csv")
    day_raw_data_path:str = os.path.join("artifacts","day_raw.csv")

    hour_train_data_path:str = os.path.join("artifacts","hour_train.csv")
    hour_test_data_path:str = os.path.join("artifacts","hour_test.csv")
    hour_raw_data_path:str = os.path.join("artifacts","hour_raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion initiated")

        try:
            df_d = pd.read_csv("notebooks\data\day.csv")
            logging.info("Read the day-dataset as dataframe")

            df_h = pd.read_csv("notebooks\data\hour.csv")
            logging.info("Read the hour-dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.day_train_data_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.hour_train_data_path),exist_ok=True)

            df_d.to_csv(self.ingestion_config.day_raw_data_path,index=False,header=True)
            logging.info("Day-Raw data saved")

            df_h.to_csv(self.ingestion_config.hour_raw_data_path,index=False,header=True)
            logging.info("Hour-Raw data saved")

            logging.info("Train test split initiated")

            day_train_set,day_test_set = train_test_split(df_d,test_size=0.2,random_state=43)
            hour_train_set,hour_test_set = train_test_split(df_h,test_size=0.2,random_state=43)

            day_train_set.to_csv(self.ingestion_config.day_train_data_path,index=False,header=True)
            day_test_set.to_csv(self.ingestion_config.day_test_data_path,index=False,header=True)
            logging.info("Train and test data of day-data is saved")

            hour_train_set.to_csv(self.ingestion_config.hour_train_data_path,index=False,header=True)
            hour_test_set.to_csv(self.ingestion_config.hour_test_data_path,index=False,header=True)
            logging.info("Train and test data of hour-data is saved")

            logging.info("Data Ingestion is completed")

            return(
                self.ingestion_config.day_train_data_path,
                self.ingestion_config.day_test_data_path,
                self.ingestion_config.hour_train_data_path,
                self.ingestion_config.hour_test_data_path
            )


        except Exception as e:
            raise CustomException(e,sys)




if __name__ == "__main__":
    obj = DataIngestion()


    obj.initiate_data_ingestion()