# Librerias

import zipfile as zf
import os
import random
import librosa
import pandas as pd
import json

# Listado de ficheros dentro de una carpeta .zip
def list_from_zip(path, zip_name):
    '''Función para saber qué ficheros tengo dentro de una carpeta .zip
    Args:
        paht (str): ruta de tu directorio. Si duda utilizas os.getcwd()
        zip_name (str): nombre de la carpeta zip
    Returns:
        list: lista de los nombres de todos los ficheros del la carpeta .zip
    '''
    zip_rute = os.path.join(path, zip_name)

    with zf.ZipFile(zip_rute, 'r') as archivos:
        list_files=archivos.namelist()
    return list_files

# Conocer el nombre de los archivos según su formato
def searching_file(path, zip_name, end):
    '''Busca las rutas de archivos de un formato concreto 
    dentro de una carpeta .zip
    Args:
        filename(str): Nombre de la carpeta .zip
        end(str): el formato del archivo que queremos guardar (e.g. .csv)
    Returns:
        list: listado del nonmbre de los archivos del mismo formato
    '''
    zip_rute = os.path.join(path, zip_name)

    with zf.ZipFile(zip_rute) as my_zip:
        my_list = []
        for file in my_zip.namelist():
            if not file.endswith(end): 
                continue
            else:
                my_list.append(os.path.join(file).replace('\\', '/'))
    return my_list

def audio_samples(path, zip_name, n):
    '''Extracción de un número concreto de archivos .ogg 
    de una carpeta tipo .zip. La carpeta donde se guardan
    siempre tiene el nombre de EDA_files.

    Args:
        paht (str): ruta de tu directorio. Si duda utilizas os.getcwd()
        zip_name (str): nombre de la carpeta zip
        n (int): numero de files a extraer

    Returns:
        list: lista del nombre de los archivos de audio seleccionados al azar
    '''
    list_names = searching_file(path, zip_name, '.ogg')
    n_sample = random.sample(list_names, n)
    print('Audio sample was created')
    return n_sample


# Abrir un archivo csv desde una carpeta comprimida
def open_csv(path, zip_name, csv_name):
    '''Lectura de archivos .csv desde una carpeta comprimida .zip
    No es necesario descomprimir los archivos para su lectura.

    Args:
        paht (str): ruta de tu directorio. Si duda utilizas os.getcwd()
        zip_name (str): nombre de la carpeta zip
        csv_name (str): nombre de la carpeta zip
    
    Returns:
        df: pandas DataFrame
    '''
    zip_folder =  zf.ZipFile(os.path.join(path, zip_name)) 
    df = pd.read_csv(zip_folder.open(csv_name))

    print('Done!')

    return df

# Abrir un archivo json desde una carpeta comprimida
def open_json(path, zip_name, json_name):
    '''Lectura de archivos .csv desde una carpeta comprimida .zip
    No es necesario descomprimir los archivos para su lectura.

    Args:
        paht (str): ruta de tu directorio. Si duda utilizas os.getcwd()
        zip_name (str): nombre de la carpeta zip
        csv_name (str): nombre de la carpeta zip
    
    Returns:
        df: pandas DataFrame
    '''
    zip_folder =  zf.ZipFile(os.path.join(path, zip_name)) 
    df = json.load(zip_folder.open(json_name))

    print('Done!')

    return df

# Extraer un número concreto de archivos de audio
def audio_samples(path, zip_name, n):
    '''Extracción de un número concreto de archivos .ogg 
    de una carpeta tipo .zip. La carpeta donde se guardan
    siempre tiene el nombre de EDA_files.

    Args:
        paht (str): ruta de tu directorio. Si duda utilizas os.getcwd()
        zip_name (str): nombre de la carpeta zip

    Returns:
        list: listado del nombres de los archivos de audio
    '''
    list_names = searching_file(path, zip_name, '.ogg')
    n_sample = random.sample(list_names, n)
    print('Audio sample was created')
    return n_sample


# Abrir una número determinado de archivos de audio desde una carpeta comprimida
def open_audio(path, zip_name, n_sample):
    '''Lectura de los archivos de audio desde una carpeta comprimida .zip
    No es necesario descomprimir los archivos para su lectura.

    Args:
        paht (str): ruta de tu directorio. Si duda utilizas os.getcwd()
        zip_name (str): nombre de la carpeta zip

    Returns:
        list: devuelve cuatro listas. La primera es la lista de las series
        temporales en 1D-numpy array. La segunda es la frecuencia de las series
        temporales. La tercera es una lista de la especie a la que corresponde
        el audio. La cuarta ruta donde está guardado el archivo.
    '''
    zip_folder = zf.ZipFile(os.path.join(path, zip_name)) 

    y_list=[] # librosa carga las series temporales en 1D-numpy array
    sr_list=[] # librosa carga la fequencia (Hz) por segundo de las series temporales
    spp=[] # guardamos la especie
    file=[] # guardamos donde se localiza el archivo de audio

    for path_ogg in n_sample:
        y, sr = librosa.load(zip_folder.open(path_ogg)) 
        y_list.append(y)
        sr_list.append(sr)
        spp.append(path_ogg.split('/')[1])
        file.append(path_ogg.split('/')[2])
    
    print('Done!')

    return y_list, sr_list, spp, file

    


    



