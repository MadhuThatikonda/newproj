import os
#import numpy as np
import flask
import pickle
from flask import Flask, render_template, request

#creating instance of the class
app=Flask(__name__)

#to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')

def ValuePredictor(to_predict_list):
    
    transform_predict=  pickle.load(open("text_vocabulary","rb"))
    to_predict = transform_predict.transform(to_predict_list).toarray()
    loaded_model = pickle.load(open("text_classifier","rb"))
    result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_query = str(request.form)
       
        to_predict_list = list(to_predict_query)
        result = ValuePredictor(to_predict_list)
            
        return render_template("result.html",result=result)
