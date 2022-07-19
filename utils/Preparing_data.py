import pickle
import os
import json
import librosa
import numpy as np
from maad import sound
from maad import util

def save_mfcc(dataset_path, n=5, n_mfcc=128, n_fft=2040,
              hop_length=512, fmin=500, fmax=8500, 
              spec_shape = (48, 128), test_duration = 5
              ):
  # Descargamos el listado de las especies objeto
  # Sólo los que estamos interesados
  scored_spp = open('/content/temp/scored_birds.json')
  scored_spp = json.load(scored_spp)

    # Diccionario para archivar los datos
  data = {
      "spp": [],
      "mfcc": []
  }

  # Loop through all the birds sound
  for i,(dirpath,dirnames,filenames) in enumerate(os.walk(dataset_path)):
      
      # Seleccionamos aquellos audios incluidos en el scored_spp
      if dirpath.split('/')[-1] in scored_spp:
          
          # Condicionamos el proceso para especies con muchas muestras
          print("\nProcessing {}".format(dirpath.split("/")[-1]))

          if len(filenames) > n:
            print(f'Too much files: processing only {n}')
          
          # Extracción de los espectrogramas
          try:
            for idx, f in enumerate(filenames): 
              if idx < n:
                
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
                  data["mfcc"].append(mfcc.tolist())
                  data["spp"].append(dirpath.split("/")[-1]) 

                print(f'Loading file number {idx}')     
              continue  
          except:
            pass
                    
  return data 