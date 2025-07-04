import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass 
from sklearn.datasets import fetch_california_housing
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts',"train.csv")
    test_data_path: str = os.path.join('artifacts',"test.csv")
    raw_data_path: str = os.path.join('artifacts',"data.csv")
    
# -------------------------------------------------------------------------
# class DataIngestionConfig:
#     def __init__(self):
#         self.train_data_path = os.path.join('artifacts',"train.csv")
#         self.test_data_path = os.path.join('artifacts',"test.csv")
#         self.raw_data_path = os.path.join('artifacts',"raw.csv")
# a = DataIngestionConfig()
# print(a.test_data_path)
# print(a.train_data_path)
# print(a.raw_data_path)
# -------------------------------------------------------------------------

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")
        try:
            housing = fetch_california_housing()
            housing_df = pd.DataFrame(housing.data,columns=housing.feature_names)
            housing_df['MedHouseVal'] = housing.target
            logging.info("fetch_california_housing Data Ingestion is Completed")
            # create an artifacts folder using make dir
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            housing_df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            # this will split the 1 array of data into 2sets, so for x data it will split into x_train,x_test similarly for y data it will split into y_train and y_test
            train_set,test_set = train_test_split(housing_df,test_size=0.3,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("splitted the data into train and test, saved the data's into thier respective paths")
            return (self.ingestion_config.train_data_path,self.ingestion_config.test_data_path)
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    di = DataIngestion()
    train_path,test_path = di.initiate_data_ingestion()
    data_transform = DataTransformation()
    train_arr,test_arr,preprocessor_obj_file_path = data_transform.initate_data_transformation(train_path,test_path)
    model = ModelTrainer()
    print(model.initiate_model_trainer(train_arr,test_arr))
