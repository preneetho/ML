#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 14:15:35 2020

@author: preneeth
"""

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import processing as fl

app = Flask(__name__) #Initialize the flask App

model = pickle.load(open('../model/HousePrediction.pkl', 'rb'))  

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    
    cols = ['cid', 'dayhours', 'room_bed', 'room_bath', 'living_measure',
       'lot_measure', 'ceil', 'coast', 'sight', 'condition', 'quality',
       'ceil_measure', 'basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long', 'living_measure15', 'lot_measure15', 'furnished',
       'total_area']
    
    df = pd.DataFrame()
    final_features=[]
    for col in cols:
        final_features.append(request.form.get(col))
        #print("cols = {} {}".format(str(col),request.form.get(col)))
        df[col]= request.form.get(col)
    
    processedData = fl.preProcessing(final_features)

    output = np.round(model.predict([processedData]),2)

    return render_template('index.html', prediction_text='House Price should be $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)