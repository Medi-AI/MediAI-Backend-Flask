#!/usr/bin/python

import os
import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

THRESHOLD = 0.05
MODEL_FILEPATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

with open(MODEL_FILEPATH, 'rb') as f:
    gbm = pickle.load(f)
    state_dict = gbm.__getstate__()
    classes_array = state_dict['classes_']
    features = state_dict['feature_names_in_']

@app.route('/', methods=['GET'])
def default():
    return jsonify({
        'message': 'Welcome to the MediAI API!'
    })

@app.route('/predict', methods=['POST'])
def predict():
    if not request.json or 'symptoms' not in request.json:
        return jsonify({
            'message': 'Invalid request, symptoms field is required'
        })

    symptoms = request.json['symptoms']
    threshold = float(request.json.get('threshold', THRESHOLD))

    if not symptoms:
        return jsonify({
            'output': [],
            'filtered_output': [],
            'message': 'No matches found'
        })
    
    coded_features = [np.where(features == keyword)[0][0] for keyword in symptoms if keyword in features]
    sample_x = np.array([i / coded_features[coded_features.index(i)] 
                         if i in coded_features else 0 
                         for i in range(len(features))]).reshape(1, -1)
    
    probs = gbm.predict_proba(sample_x.reshape(1, -1))[0]
    output = np.column_stack((classes_array, probs.astype(float)))

    filtered_output = output[probs > threshold]
    filtered_output[:, 1] = filtered_output[:, 1].astype(float)

    output = output[np.argsort(output[:, 1])[::-1]]
    filtered_output = filtered_output[np.argsort(filtered_output[:, 1])[::-1]]

    if len(filtered_output) == 0:
        filtered_output = output[:1]
    
    return jsonify({
        'output': output.tolist(),
        'filtered_output': filtered_output.tolist(),
        'message': f'{len(filtered_output)} matches found'
    })

@app.route('/metadata', methods=['GET'])
def metadata():
    return jsonify({
        'classes': classes_array.tolist(),
        'features': features.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
