from webapp import app

from flask import request
import tensorflow as tf
import numpy as np
import pandas as pd

from webapp.mnist_model import MNIST_Model

@app.route("/")
def index():
  return "Hello World!"

@app.route("/mnist_model", methods=["GET"])
def mnist_model():
  image = request.args.get('image')
  if image:
    return image
  y = MNIST_Model.predict_digit(3)
  return f"hello {y}"