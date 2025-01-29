from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__,static_folder='static')

app=application

from datetime import datetime

def calculate_age(birth_date):
    # Convert the input string to a datetime object
    birth_date = datetime.strptime(birth_date, "%d/%m/%Y")
    today = datetime.today()
    age = today.year - birth_date.year
    if today.month < birth_date.month or (today.month == birth_date.month and today.day < birth_date.day):
        age -= 1
    return age
  
## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html',results="Submit")
    else:
        data=CustomData(
            age=int(calculate_age(request.form.get('birthday'))),
            occupation=request.form.get('Occupation'),
            annual_income=int(request.form.get('Annual Income')),
            monthly_inhand_salary=float(request.form.get('Monthly Inhand Salary')),
            num_of_bank_accounts=int(request.form.get('Number of Bank Accounts')),
            num_of_credit_cards=int(request.form.get('Number of Credit Cards')),

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0][0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0", debug = False)        


