from webapp import app

from flask import json, request, jsonify
import tensorflow as tf
import numpy as np

from webapp.mnist_model import MNIST_Model

@app.route("/")
def index():
  return "Hello World!"

@app.route("/api/mnist_model", methods=["POST"])
def mnist_model():
  image = request.get_json()
  try:
    pred = MNIST_Model.predict_digit(image)
    return jsonify({
      "prediction": pred,
    })
  except:
    return jsonify({
      "error": "There has been an error",
    })