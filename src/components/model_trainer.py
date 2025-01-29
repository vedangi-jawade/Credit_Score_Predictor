import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                #"Random Forest": RandomForestClassifier(),
                #"Decision Tree": DecisionTreeClassifier(),
                #"Gradient Boosting": GradientBoostingClassifier(),
                #"XGBClassifier": XGBClassifier(),
                "CatBoosting Classifier": CatBoostClassifier(verbose=False),
                #"AdaBoost Classifier": AdaBoostClassifier(),
            }
            params={
                #"Decision Tree": {
                 #   'criterion':["gini", "entropy", "log_loss"],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                #},
                #"Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                 #   'n_estimators': [8,16,32,64,128,256]
                #},
                #"Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                 #   'learning_rate':[.1,.01,.05,.001],
                  #  'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                   # 'n_estimators': [8,16,32,64,128,256]
                #},
                #"XGBClassifier":{
                 #   'learning_rate':[.1,.01,.05,.001],
                  #  'n_estimators': [8,16,32,64,128,256]
                #},
                "CatBoosting Classifier":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                #"AdaBoost Classifier":{
                 #   'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                  #  'n_estimators': [8,16,32,64,128,256]
                #}
                
            }
            print("started with training")
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            print("training ended")
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            print(model_report)
            best_model = models[best_model_name]
            print("Got the best model",best_model_score)
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = classification_report(y_test, predicted)
            return r2_square
            



            
        except Exception as e:
            raise CustomException(e,sys)