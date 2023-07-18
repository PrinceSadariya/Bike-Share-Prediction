import os
import sys
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    day_c_model_path = os.path.join("artifacts","day_c_model.pkl")
    hour_c_model_path = os.path.join("artifacts","hour_c_model.pkl")
    day_r_model_path = os.path.join("artifacts","day_r_model.pkl")
    hour_r_model_path = os.path.join("artifacts","hour_r_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,day_train_df,day_test_df,hour_train_df,hour_test_df):
        try:
            logging.info("Splitting train and test input data")

            X_c_day_train , y_c_day_train , X_c_day_test , y_c_day_test = (
                day_train_df.iloc[:,:8],
                day_train_df.iloc[:,-2],
                day_test_df.iloc[:,:8],
                day_test_df.iloc[:,-2]
            )
            X_r_day_train , y_r_day_train , X_r_day_test , y_r_day_test = (
                day_train_df.iloc[:,:8],
                day_train_df.iloc[:,-1],
                day_test_df.iloc[:,:8],
                day_test_df.iloc[:,-1]
            )
            X_c_hour_train , y_c_hour_train , X_c_hour_test , y_c_hour_test = (
                hour_train_df.iloc[:,:9],
                hour_train_df.iloc[:,-2],
                hour_test_df.iloc[:,:9],
                hour_test_df.iloc[:,-2]
            )
            X_r_hour_train , y_r_hour_train , X_r_hour_test , y_r_hour_test = (
                hour_train_df.iloc[:,:9],
                hour_train_df.iloc[:,-1],
                hour_test_df.iloc[:,:9],
                hour_test_df.iloc[:,-1]
            )

            logging.info("splitting of input feature and target feature is completed")

            day_c_model = GradientBoostingRegressor(max_features='sqrt',n_estimators=128,subsample=0.6)
            day_r_model = GradientBoostingRegressor(max_features='sqrt',n_estimators=128,subsample=0.6)
            hour_c_model = GradientBoostingRegressor(max_features='sqrt',n_estimators=128,subsample=0.6)
            hour_r_model = GradientBoostingRegressor(max_features='sqrt',n_estimators=128,subsample=0.6)

            logging.info("Model training on training data is started")

            day_c_model.fit(X_c_day_train,y_c_day_train)
            day_r_model.fit(X_r_day_train,y_r_day_train)
            hour_c_model.fit(X_c_hour_train,y_c_hour_train)
            hour_r_model.fit(X_r_hour_train,y_r_hour_train)

            logging.info("Model training on train data is completed")

            save_object(self.model_trainer_config.day_c_model_path,day_c_model)
            save_object(self.model_trainer_config.day_r_model_path,day_r_model)
            save_object(self.model_trainer_config.hour_c_model_path,hour_c_model)
            save_object(self.model_trainer_config.hour_r_model_path,hour_r_model)

            logging.info("All models are saved ")

            day_y_c_pred = day_c_model.predict(X_c_day_test)
            day_y_r_pred = day_r_model.predict(X_r_day_test)
            hour_y_c_pred = hour_c_model.predict(X_c_hour_test)
            hour_y_r_pred = hour_r_model.predict(X_r_hour_test)

            day_c_model_score = r2_score(y_c_day_test,day_y_c_pred)
            day_r_model_score = r2_score(y_r_day_test,day_y_r_pred)
            hour_c_model_score = r2_score(y_c_hour_test,hour_y_c_pred)
            hour_r_model_score = r2_score(y_r_hour_test,hour_y_r_pred)

            return(day_c_model_score,day_r_model_score,hour_c_model_score,hour_r_model_score)

        except Exception as e:
            raise CustomException(e,sys)