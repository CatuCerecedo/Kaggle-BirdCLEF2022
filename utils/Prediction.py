# Librerias

import os
import librosa
import numpy as np
from maad import sound
from maad import util

# Función para cargas los datos a predecir
def save_mfcc_test(dataset_path,n=5, n_mfcc=128, n_fft=2040,
              hop_length=512, fmin=500, fmax=8500):
   
  # Diccionario para archivar los datos
  data = []
  # Loop through all the birds sound
  for i,(dirpath,dirnames,filenames) in enumerate(os.walk(dataset_path)):

    for idx, f in enumerate(filenames):
             
      # Cargamos el archivo de audio
      file_path = os.path.join(dirpath,f)
      signal,sr = librosa.load(file_path)

      # Eliminamos las zonas de silencio
      signal,index = librosa.effects.trim(signal)
      
      # Dividimos el audio en secciones para poder procesar
      stream = list(librosa.stream(file_path,
            block_length=256,
            frame_length=4096,
            hop_length=1024))
      
      for y_block in stream[:-1]:

        mfcc = librosa.feature.mfcc(y_block,
                                  sr =sr,
                                  n_fft=n_fft,
                                  hop_length=hop_length,
                                  fmin=fmin,
                                  fmax=fmax)
        

        rm_bk, noise_profile, _=sound.remove_background(mfcc, gauss_win=100)
        mfcc = librosa.power_to_db(rm_bk, ref=np.max).astype(np.float32)

        mfcc = mfcc.T

        # Guardamos los datos
        data.append(mfcc.tolist())
      print(f'Loading file number {idx}')
  data = np.array(data)
  return data 