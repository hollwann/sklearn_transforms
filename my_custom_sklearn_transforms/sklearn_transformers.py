from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from tensorflow import keras
# All sklearn Transforms must have the `transform` and `fit` methods


b = np.array([[0., 0.],
              [7712736.230862944, 2648986.7236121534],
              [0., 0.],
              [0.5066484907497566, 0.4772438497094733],
              [0.10954235637779941, 0.3123184729358777],
              [0.2650113599480688, 0.44133925618116493],
              [0.21648815319701395, 0.41185074083017037],
              [0.13209996754300551, 0.33859941836651525],
              [116.82929557309915, 4819.851155007611],
              [0.035430785966569296, 1.7068736563691411],
              [-0.03880460782018825, 1.7276359792696574],
              [4409.2679513727535, 27667.76093396526],
              [1.205419434696527, 15.834994945837217],
              [-1.0491985980688088, 13.29265749256275],
              [541.5253667640377, 674.4431085832614],
              [354.5389045764362, 1049.1888484562153],
              [-253.73866439467704, 410.12452630022125],
              [2466.690184915612, 4180.01632543064],
              [130.4127991837066, 188.73294736312062],
              [-130.25218583414474, 188.81174813929428],
              [24747.386481661804, 79319.48963248247],
              [61.09623498864005, 195.2151937851121],
              [-63.243362544628376, 251.13082901865621],
              [28.15259006815969, 348.81817645555833],
              [6.975491723466407, 122.68282221337365],
              [-7.220813047711781, 149.2006229345841],
              [1130.7700421940929, 821.6600264999933],
              [8595.691374553717, 188047.23840702316],
              [3294.859990262902, 55773.56483495419],
              [-3694.0336660175267, 77743.56356612715],
              [300.45271015903927, 869.9227162657572],
              [1.2388834793898085, 0.6448512176323228],
              [5690.877150275885, 783.0228592563196],
              [142.27004219409284, 48.16748788304037],
              [-158.915611814346, 74.3735976121195],
              [3188.207785296982, 1945.5549833371558],
              [72.63205371632587, 120.77826232506195],
              [-67.39823028237585, 81.17609474798572],
              [1539.6267624148004, 6024.387792474067],
              [281.1542713404739, 804.095594492356],
              [-320.7899383317105, 1441.0362225273948],
              [3136.010530991561, 28684.132634598896],
              [120.63631108714702, 1820.7133709780132],
              [14297.486349848332, 1336.226185218817]])


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


def to_one_hot(arr):
    output = arr.astype(int)
    b = np.zeros((output.size, output.max()+1))
    b[np.arange(output.size), output] = 1
    return b


def class_to_one_hot(arr, remove_first=True):
    genero = np.array(arr)
    genero[genero == '0'] = ''
    _, generoid = np.unique(genero, return_inverse=True)
    generooh = to_one_hot(generoid)
    return generooh


def freq_norm(arr, mean, var):
    edad = np.array(arr)
    edad[edad == ''] = mean
    nan_array = (arr == '').astype(float)
    edad = (edad.astype(float) - mean) / var
    return np.hstack((edad[:, np.newaxis], nan_array[:, np.newaxis]))


class Normalize():
    def fit(self, X, y=None):
        return self

    def predict(self, X, y=None):
        return X

    def transform(self, X):
        csv = np.array(X, dtype=str)
        csv[pd.isnull(csv)] = ''
        data = np.zeros((csv.shape[0], 0))
        for col in range(1, 44):
            if col == 2:
                # continue
                u = np.array(['', 'CANDIDATE', 'FALSE POSITIVE'])
                column = class_to_one_hot(csv[:, col])
            elif col == 3:
                edad = np.array(csv[:, 3])
                mean = b[3, 0]
                var = b[3, 1]
                edad[edad == ''] = mean
                nan_array = (csv[:, 3] == '').astype(float)
                edad = edad.astype(float)
                edad[edad > 1.0] = edad[edad > 1.0] / 1000.0
                edad = (edad.astype(float) - mean) / var
                column = np.hstack(
                    (edad[:, np.newaxis], nan_array[:, np.newaxis]))
            elif col == 43:
                edad = np.array(csv[:, col])
                mean = b[col, 0]
                var = b[col, 1]
                edad[edad == ''] = mean
                nan_array = (csv[:, col] == '').astype(float)
                edad = edad.astype(float)
                edad[edad < 20.0] = edad[edad < 20.0] * 1000.0
                edad = (edad.astype(float) - mean) / var
                column = np.hstack(
                    (edad[:, np.newaxis], nan_array[:, np.newaxis]))
            else:
                column = freq_norm(csv[:, col], b[col, 0], b[col, 1])

            data = np.hstack((data, column))

        return data  # data[(np.abs(data) <=2).all(1) ]


# modelo de la red
def build_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(64, activation='relu', input_shape=(87,)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(3, activation='softmax'))

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
        labels = np.array(['0', '1', '2'], dtype='<U23')

        return labels[prediction]

    def predict_proba(self, x, y=None):
        model = build_model()
        model.set_weights(self.weights)

        a = np.argmax(model.predict(x), axis=1)
        b = np.zeros((a.size, 3))
        b[np.arange(a.size), a] = 1

        return b
