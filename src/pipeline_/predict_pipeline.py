import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        annual_income: int,
        monthly_inhand_salary : int,
        num_of_bank_accounts:int,
        num_of_credit_cards: int,
        age: int,
        occupation : str):

        self.annual_income = annual_income

        self.monthly_inhand_salary = monthly_inhand_salary

        self.num_of_bank_accounts = num_of_bank_accounts

        self.num_of_credit_cards = num_of_credit_cards

        self.age = age

        self.occupation = occupation

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Annual_Income": [self.annual_income],
                "Monthly_Inhand_Salary": [self.monthly_inhand_salary],
                "Num_Bank_Accounts": [self.num_of_bank_accounts],
                "Num_Credit_Card": [self.num_of_credit_cards],
                "Age": [self.age],
                "Occupation": [self.occupation],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

