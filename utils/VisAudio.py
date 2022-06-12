# Librerias

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from maad import sound
from maad import util
import IPython.display as ipd

# Oscilograma mediante scikit-maad
def sm_waveplot(y_list, sr_list, spp, sound=True):
    '''Grafica un oscilograma

    Args:
        y_list (list): lista de las series temporales de audio en 1D-numpy
        sr_list (list): lista de las frecuencuas de audio de las series temporales
        spp (list): lista de las especies

        ATENCIÓN: todas las listas tienen que tener el mismo número de elementos
    '''
    for i in range(len(sr_list)):
        util.plot_wave(y_list[i], sr_list[i], figtitle=spp[i])
        if sound==True:
            ipd.display(ipd.Audio(y_list[i], rate=sr_list[i]))
        else:
            pass

# Espectrograma mediante el módulo librosa
def lib_spec(y_list, sr_list, spp, db=False):
    '''Grafica un espectrograma con el módulo librosa

    Args: 
        y_list (list): lista de las series temporales de audio en 1D-numpy
        sr_list (list): lista de las frecuencuas de audio de las series temporales
        spp (list): lista de las especies
        db (bool, optional): Si es True también grafica el espectrograma sin pasar los herzios a decibelios.
        Defaults False.
    '''
    for i in range(len(sr_list)):
        #Step 1. Transformamos la serie temporal mediante STFT
        stft = librosa.stft(y_list[i])
        if db==True:
            plt.colorbar(librosa.display.specshow(stft,sr=sr_list[i],x_axis='time',y_axis='hz'))
            plt.title('Spectrogram of ' + spp[i])
            plt.show()
        else:
            pass
        # Step 2. Dibujamos el espectrograma´
        stft_db=librosa.amplitude_to_db(abs(stft))
        plt.colorbar(librosa.display.specshow(stft_db,sr=sr_list[i],x_axis='time',y_axis='hz'))
        plt.title('Spectrogram of ' + spp[i])
        plt.show()

def sm_espec(y_list, sr_list, spp, N=1024):
    '''Grafica un espectrograma con el módulo scikit-maad
    Args:
        y_list (list): lista de las series temporales de audio en 1D-numpy
        sr_list (list): lista de las frecuencuas de audio de las series temporales
        spp (list): lista de las especies
        N (int, optional): Longitud del segmento para calcular la trasformación de Fourier.
        Mirar en la ayuda de skit-maad. Defaults 1024
    '''
    for i in range(len(sr_list)):
        # Condición para conservar los picos de energía necesarios para la clasificación
        # Step 1. Función para la tranformación STFT
        y_power,tn,fn,ext=sound.spectrogram(y_list[i], sr_list[i], mode = 'psd') 
        # Step 2. Convertir a db
        y_dB=util.power2dB(y_power, db_range=96) + 96 
        # Step 3. Graficar
        fig_kwargs = {'vmax': np.median(y_dB)+40,
                            'vmin':np.median(y_dB),
                            'extent':ext,
                            'figsize':(4,13),
                            'title':'Power spectrogram density (PSD) ' + spp[i],
                            'xlabel':'Time [sec]',
                            'ylabel':'Frequency [Hz]',
                            }
        fig, ax = util.plot2d(y_dB,**fig_kwargs)     