import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import sys

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path = "artifacts\model.pkl"
            preprocessor_path = "artifacts\preprocessor.pkl"
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    # this function is responsible for to map the data which we were giving at frontend to backend HTML.
    def __init__(self,MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude):
        self.MedInc = MedInc
        self.HouseAge = HouseAge
        self.AveRooms = AveRooms
        self.AveBedrms = AveBedrms
        self.Population = Population
        self.AveOccup = AveOccup
        self.Latitude = Latitude
        self.Longitude = Longitude

    def get_data_as_frame(self):
        try:
            custom_data_input_dict = {
                "MedInc":[self.MedInc],
                "HouseAge":[self.HouseAge],
                "AveRooms":[self.AveRooms],
                "AveBedrms":[self.AveBedrms],
                "Population":[self.Population],
                "AveOccup":[self.AveOccup],
                "Latitude":[self.Latitude],
                "Longitude":[self.Longitude],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)
    
    