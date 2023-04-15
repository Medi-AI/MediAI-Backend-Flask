#!/usr/bin/python

import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)
THRESHOLD = 0.2

with open('model.pkl', 'rb') as f:
    gbm = pickle.load(f)
    state_dict = gbm.__getstate__()
    classes_array = state_dict['classes_']
    features = state_dict['feature_names_in_']

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.json.get('query', [])
    threshold = float(request.json.get('threshold', THRESHOLD))
    if not symptoms:
        return jsonify({
            'output': [],
            'filtered_output': [],
            'message': 'No matches found'
        })
    
    coded_features = [np.where(features == keyword)[0][0] for keyword in symptoms]
    sample_x = np.array([i / coded_features[coded_features.index(i)] 
                         if i in coded_features else 0 
                         for i in range(len(features))]).reshape(1, -1)
    
    probs = gbm.predict_proba(sample_x.reshape(1, -1))[0]
    output = np.column_stack((classes_array, probs.astype(float)))

    filtered_output = output[probs > threshold]
    filtered_output[:, 1] = filtered_output[:, 1].astype(float)

    output = output[np.argsort(output[:, 1])[::-1]]
    filtered_output = filtered_output[np.argsort(filtered_output[:, 1])[::-1]]
    
    return jsonify({
        'output': output.tolist(),
        'filtered_output': filtered_output.tolist(),
        'message': f'{len(filtered_output)} Matches found'
    })

@app.route('/metadata', methods=['GET'])
def metadata():
    return jsonify({
        'classes': classes_array.tolist(),
        'features': features.tolist()
    })

# if __name__ == '__main__':
#     app.run(debug=True)
