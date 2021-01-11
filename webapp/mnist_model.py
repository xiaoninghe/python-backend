from os.path import join
import numpy as np
import tensorflow as tf

class MNIST_Model:

    model = tf.keras.models.load_model(join('webapp', 'models', 'mnist_model'))

    @classmethod
    def predict_digit(cls, X):
        pred = cls.model.predict(X)
        pred = np.argmax(pred, axis=1)
        return pred[0]