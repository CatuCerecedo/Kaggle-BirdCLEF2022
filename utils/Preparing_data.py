import json
import os
import librosa
from maad import sound
from maad import util

SAVE_JSON = 'data_espec.json'
dataset_json = '/content/temp/scored_birds.json'

def save_espec(dataset_path, **dataset_json, db = 96, n = 96):
  scored_spp = open(dataset_json)
  scored_spp = json.load(scored_spp)

  # Guardamos los datos en listas
  spp = []
  mfcc = []

  # Creamos un loop para cargar los archivos
  for dirpath, dirnames, filenames in os.walk(dataset_path):

    # Aseguramos que guardamos las especies que nos interesan
    if dirpath.split('/')[-1] in scored_spp:

      print("\nProcessing: {}".format(dirpath.split("/")[-1]))

      for f in filenames:
        # Cargamos el audio
        file_path = os.path.join(dirpath, f)
        y, sr = librosa.load(file_path)

        # Generamos los espectrogramas
        y_power,tn,fn,ext=sound.spectrogram(y, sr, method='psd') 

        # Procesado del audio, limpiamos los sonidos de fondo
        y_dB=util.power2dB(y_power, db_range=db) + n
        rm_bk, noise_profile, _=sound.remove_background(y_dB)

        # Guardamos el array del espectrograma
        mfcc.append(rm_bk)
      
        # Guardamos nombre spp
        spp.append(dirpath.split('/')[-1])

  data = {
    'spp': spp,
    'mfcc': mfcc
  }

  # Save object un json file
  with open(SAVE_JSON, "w") as fp:
    json.dump(data, fp, indent=4)