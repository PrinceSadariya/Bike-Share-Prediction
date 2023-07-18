import sys
import pandas as pd
from src.exception import CustomException
from datetime import datetime
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            day_c_model_path = "artifacts\day_c_model.pkl"
            day_r_model_path = "artifacts\day_r_model.pkl"
            hour_c_model_path = "artifacts\hour_c_model.pkl"
            hour_r_model_path = "artifacts\hour_r_model.pkl"

            day_c_model = load_object(day_c_model_path)
            day_r_model = load_object(day_r_model_path)
            hour_c_model = load_object(hour_c_model_path)
            hour_r_model = load_object(hour_r_model_path)

            if "hr" in features.columns:
                hour_c_pred = hour_c_model.predict(features)
                hour_r_pred = hour_r_model.predict(features)

                return[hour_c_pred,hour_r_pred]

            else:
                day_c_pred = day_c_model.predict(features)
                day_r_pred = day_r_model.predict(features)

                return[day_c_pred,day_r_pred]

        except Exception as e:
            raise CustomException(e,sys)




class CustomData:
    def __init__(
        self,
        date,
        hour,
        holiday,
        weathersit,
        temp,
        humidity,
        windspeed
    ):
        
        self.hr = int(hour)
        self.holiday = int(holiday)
        self.weathersit = int(weathersit)
        self.temp = float(temp) / 41
        self.hum = float(humidity) / 100
        self.windspeed = float(windspeed) / 67


        # for finding season

        # 21-11 to 20-02 ==> season 1
        # 21-01 to 20-05 ==> season 2
        # 21-05 to 20-08 ==> season 3
        # 21-08 to 20-11 ==> season 4

        year = int(date.split("-")[0])
        new_date = datetime.strptime(date,"%Y-%m-%d")

        if new_date <= datetime(year,3,20):
            self.season = 1
        elif datetime(year,3,21) <= new_date <= datetime(year,6,20):
            self.season = 2
        elif datetime(year,6,21) <= new_date <= datetime(year,9,20):
            self.season = 3
        elif datetime(year,9,21) <= new_date <= datetime(year,12,20):
            self.season = 4
        else:
            self.season = 1


        weekday = new_date.weekday() + 1

        if weekday == 7:
            weekday = 0
        
        self.weekday = weekday

        # for finding day is working day or not

        if self.holiday == 1 or weekday == 0 or weekday == 6:
            self.workingday = 0
        else:
            self.workingday = 1
        


    def get_data_as_data_frame(self):
        try:
            if self.hr == 24:
                custom_data_dict = {
                   "season":[self.season],
                    "holiday":[self.holiday],
                    "weekday":[self.weekday],
                    "workingday":[self.workingday],
                    "weathersit":[self.weathersit],
                    "temp":[self.temp],
                    "hum":[self.hum],
                    "windspeed":[self.windspeed]
                }

                return pd.DataFrame(custom_data_dict)

            else:
                custom_data_dict = {
                    "season":[self.season],
                    "hr":[self.hr],
                    "holiday":[self.holiday],
                    "weekday":[self.weekday],
                    "workingday":[self.workingday],
                    "weathersit":[self.weathersit],
                    "temp":[self.temp],
                    "hum":[self.hum],
                    "windspeed":[self.windspeed]
                }

                return pd.DataFrame(custom_data_dict)

        except Exception as e:
            raise CustomException(e,sys)
