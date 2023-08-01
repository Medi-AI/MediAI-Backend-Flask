#!/usr/bin/python

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

MODEL_FILEPATH = "model.pkl"

with open(MODEL_FILEPATH, "rb") as f:
    gbm = pickle.load(f)
    state_dict = gbm.__getstate__()
    classes_array = state_dict["classes_"]
    features_dict = {
        feature: i for i, feature in enumerate(state_dict["feature_names_in_"])
    }


@app.route("/", methods=["GET"])
def default():
    return jsonify({"message": "Welcome to the MediAI API!"})


@app.route("/predict", methods=["POST"])
def predict():
    if not request.json or "symptoms" not in request.json:
        return jsonify({"message": "Invalid request, symptoms field is required"})

    symptoms = request.json["symptoms"]

    if not symptoms:
        return jsonify({"output": []})

    coded_features = [
        features_dict[keyword] for keyword in symptoms if keyword in features_dict
    ]

    sample_x = np.zeros(len(features_dict), dtype=np.float32)
    sample_x[coded_features] = 1

    probs = gbm.predict_proba(sample_x.reshape(1, -1))[0]

    output = list(zip(classes_array, probs.astype(float)))
    output.sort(key=lambda x: x[1], reverse=True)

    return jsonify({"output": output[:3]})


@app.route("/metadata", methods=["GET"])
def metadata():
    return jsonify({"classes": classes_array.tolist(), "features": list(features_dict)})
