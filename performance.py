# Librerias
import pickle
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from utils.Preparing_data import save_mfcc_test
from utils.VisAudio import sm_waveplot

import librosa

# Descargamos el modelo y el LabelEncoder

model = new_model = tf.keras.models.load_model('Outputs\my_model')
y_encoder = open('Outputs\y_encoder.pickle','rb')
y_encoder = pickle.load(y_encoder)

# Procesamiento de datos de entreada
file_path = 'soundscape.ogg'
spp = ['soundscape.ogg']

signal, sr = librosa.load(file_path)
sm_waveplot([signal], [sr], spp, sound=True)

# Predicción del Audio
n=5
n_mfcc=128
n_fft=2040 
hop_length=512

test_data = save_mfcc_test(file_path,
                 n=n,
                 n_mfcc=n_mfcc,
                 n_fft=n_fft,
                 hop_length=hop_length) 

# Predicción
prediction = model.predict(test_data)

predicted_index = np.argmax(prediction, axis=1)

y = y_encoder.inverse_transform(predicted_index)

print("\nOur results are:\nNumber of audio segments: {}\nPredicted label: {}".format(len(y), y))

