# Librerias

# Gestion base de datos
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Creación redes neuronales
from keras.models import Sequential
from keras.layers import Dense, Dropout, MaxPooling2D, LSTM

import matplotlib.pyplot as plt

# Definimos el formato de las variables x,y
def fromat_xy(x,y):
  # Tranformamos lista a array
  x = np.array(x)

  # Label Encoder de la variable respuesta
  y_encoder = LabelEncoder()
  y_encoder.fit(y)
  y = y_encoder.transform(y)

  return x,y,y_encoder

# Función para obtener el train, validation y test split
def data_split (x,y, test_size, validation_size):
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size,
   random_state = 1234567)
  x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, 
  test_size=validation_size, random_state = 1234567)

  return x_train, x_test, y_train, y_test, x_validation, y_validation


def build_model(input_shape):

  # Genramos la capa de emtrada
  model = Sequential()

  # 2 capas recurrentes LSTM
  model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
  model.add(LSTM(64))
  model.add(Dropout(0.1))

  # Una capa oculta
  model.add(Dense(32, activation='relu'))
  model.add(Dropout(0.3))

  # Capa de salida
  model.add(Dense(21, activation='softmax'))

  return model


# Función para evluar el entrenamiento del modelo
def plot_history(history, save):

  # Gráficos de la accuracy y el loss del modelo
  fig, axs = plt.subplots(1,2, figsize=(15, 5))

  # Creamos el subplot de la accuracy
  axs[0].plot(history.history["accuracy"], 'co', label="Training accuracy")
  axs[0].plot(history.history["val_accuracy"], 'm', label="Validation accuracy")
  axs[0].set_ylabel("Accuracy")
  axs[0].set_xlabel("Epochs")
  axs[0].set_title("Training and validation accuracy")
  axs[0].legend(loc = 'lower right')

  # Creamos el subplot de validación loss
  axs[1].plot(history.history["loss"], 'co', label="Training loss")
  axs[1].plot(history.history["val_loss"], 'm', label="Validation loss")
  axs[1].set_ylabel("Loss")
  axs[1].set_xlabel("Epochs")
  axs[1].set_title("Training and validation loss")
  axs[1].legend(loc = 'lower left')

  plt.show()

  # Dpendiendo de si se quiere guardar
  if save == False:
      pass
  else:
      fig.savefig('Artist_mean_barplots.png', facecolor='white')