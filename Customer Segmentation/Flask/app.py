import numpy as np
import pickle
import pandas
import os
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = pickle.load(open('C:/Users/hp/Desktop/Customer_Segmentation/cust_xgbmodel.pkl','rb'))
     
@app.route('/')
def home():
    return render_template('index.html')                  

@app.route('/predict', methods = ["POST", "GET"]) 
def predict():
    input_feature = [float(x) for x in request.form.values()]
    features_values = [np.array(input_feature)]

    names = [['Sex', 'MaritalStatus', 'Age', 'Education', 'Income', 'Occupation', 'SettlementSize']]    

    data = pandas.DataFrame(features_values, columns = names)

    prediction = model.predict(data)
    print(prediction)
                                                                                       
    if(prediction == 0):
        return render_template("index.html", prediction_text = "Not a potential customer")
    elif(prediction == 1):
        return render_template("index.html", prediction_text = "Potential customer")
    else:
        return render_template("index.html", prediction_text = "Highely Potential customer")

if __name__=="__main__":
    port = int(os.environ.get('PORT',5000))
    app.run(port = port, debug = True, use_reloader = False) 