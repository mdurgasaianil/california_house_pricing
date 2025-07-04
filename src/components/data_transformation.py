import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from dataclasses import dataclass 
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transfomation_config = DataTransformationConfig()
    
    def get_data_transformation_config(self):
        try:
            num_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup','Latitude', 'Longitude']
            num_pipeline = Pipeline(
                steps=[
                    ("scaler",StandardScaler())
                ]
            )
            preprocessor = ColumnTransformer(
            transformers=[
                    ('num_pipeline',num_pipeline, num_features),
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    def initate_data_transformation(self,train_path,test_path):
        try:
            logging.info('Intiated data transformation')
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocessor_obj = self.get_data_transformation_config()
            target_column_name="MedHouseVal"

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            save_object(
                file_path=self.data_transfomation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transfomation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)