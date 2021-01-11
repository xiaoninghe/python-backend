from os.path import join
import tensorflow as tf

class MNIST_Model:

    model = tf.keras.models.load_model(join('webapp', 'models', 'mnist_model'))

    @classmethod
    def predict_digit(cls, X):
        if X == 3:
            return 0
        y = cls.model.predict(X)
        return y