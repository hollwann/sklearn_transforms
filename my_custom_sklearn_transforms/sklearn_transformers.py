from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from tensorflow import keras
# All sklearn Transforms must have the `transform` and `fit` methods


class DropColumns():
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def predict(self, X, y=None):
        return X

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return np.array(data.drop(labels=self.columns, axis='columns'))

# All sklearn Transforms must have the `transform` and `fit` methods


class Normalize():
    def fit(self, X, y=None):
        return self

    def predict(self, X, y=None):
        return X

    def transform(self, X):
        x = np.array(X, dtype=float)
        x[pd.isnull(x)] = 0.

        x_max = np.array([127., 147., 124.,  12.,  12.,  13.,  13.,  14.,  12., 100., 100.,
                          100.])

        x_norm = np.divide(x, x_max)

        x_norm[pd.isnull(X)] = 0.
        return x_norm


# modelo de la red
def build_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(256, activation='relu', input_shape=(12,)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(6, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


class Keras():
    def __init__(self, weights):
        self.weights = weights

    def fit(self, X, y=None):
        return self

    def transform(self, x):
        return x

    def predict(self, x, y=None):
        model = build_model()
        model.set_weights(self.weights)

        prediction = np.argmax(model.predict(x), axis=1)
        labels = np.array(['advanced_backend', 'advanced_data_science', 'advanced_front_end',
                           'beginner_backend', 'beginner_data_science', 'beginner_front_end'], dtype='<U23')

        return labels[prediction]

    def predict_proba(self, x, y=None):
        model = build_model()
        model.set_weights(self.weights)

        a = np.argmax(model.predict(x), axis=1)
        b = np.zeros((a.size, 6))
        b[np.arange(a.size), a] = 1

        return b
