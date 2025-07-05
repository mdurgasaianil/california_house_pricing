import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route("/predict_api",methods=['POST'])
def predict_api():
    if request.method =="GET":
        return render_template('home.html')
    else:
        data = request.json['data']
        custom_data = CustomData(
            MedInc=data['MedInc'],
            HouseAge=data['HouseAge'],
            AveRooms=data['AveRooms'],
            AveBedrms=data['AveBedrms'],
            Population=data['Population'],
            AveOccup=data['AveOccup'],
            Latitude=data['Latitude'],
            Longitude=data['Longitude']
        )
        pred_df = custom_data.get_data_as_frame()
        print(pred_df)
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        print(results[0])
        return str(results[0])
    
@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=="GET":
        return render_template('home.html')
    else:
        custom_data = CustomData(
            MedInc=request.form.get('MedInc'),
            HouseAge=request.form.get('HouseAge'),
            AveRooms=request.form.get('AveRooms'),
            AveBedrms=request.form.get('AveBedrms'),
            Population=request.form.get('Population'),
            AveOccup=request.form.get('AveOccup'),
            Latitude=request.form.get('Latitude'),
            Longitude=request.form.get('Longitude')
        )
        pred_df = custom_data.get_data_as_frame()
        print(pred_df)
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        print(results[0])
        return render_template("home.html",prediction_text = "The house price prediction is ${:.2f}k".format(results[0]))

if __name__ == '__main__':
    app.run(debug=True)
