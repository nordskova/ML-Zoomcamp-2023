import pickle
import numpy as np
from flask import Flask, request, jsonify

with open('model2.bin', 'rb') as f_in: 
    model = pickle.load(f_in)
with open('dv.bin', 'rb') as f_in: 
    dv = pickle.load(f_in)

def predict_single(client, dv, model):
    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0,1]
    return y_pred

app = Flask('credit')

@app.route('/predict', methods=['POST'])  ## in order to send the customer information we need to post its data.
def predict():
    client = request.get_json()  ## web services work best with json frame, So after the user post its data in json format we need to access the body of json.

    prediction = predict_single(client, dv, model)
    churn = prediction >= 0.5

    result = {
    'credit_probability': float(prediction), ## we need to cast numpy float type to python native float type
    'credit': bool(churn),  ## same as the line above, casting the value using bool method
    }

    return jsonify(result) 

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)

