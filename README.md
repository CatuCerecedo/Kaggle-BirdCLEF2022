# FinalPoject
Final project ID Bootcamp Data Science and Machine Learning

{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Kaggle competition: BirdCLEF 2022\n",
    "\n",
    "**Objetivo**: identificar cantos de aves mediante grabaciones en campo abierto\n",
    "\n",
    "**Descripción**\n",
    "\n",
    "(Resumen de la descripción de [Kaggle](https://www.kaggle.com/competitions/birdclef-2022/overview))\n",
    "\n",
    "Hawaii ha perdido el 68 % de sus especies de aves alterando otros procesos ecológicos fundamentales (e.g., la alteración de las cadenas de alimento). Los investigadores suelen utilizar métodos de captura directa para la monitorización de poblaciones salvajes, i.e., la captura, marcajae y liberación de individuos. Estas herramientas de seguimiento permiten comprender cómo una población reaccionan a los cambios en el medio ambiente. Desafortunadamente, muchas de las aves de este archipiélago están aisladas en hábitats elevados y de difícil acceso. Estas condiciones dificultan el monitoreo físico, provocando que los científicos recurran a las grabaciones de sonido, también conocido como bioseguimiento o monitoreo bioacústico. Este enfoque podría proporcionar una estrategia pasiva, de bajo costo y rentable para estudiar poblaciones de aves en peligro de extinción.\n",
    "\n",
    "Los métodos actuales para procesar grandes conjuntos de datos bioacústicos implican la anotación manual de cada grabación. Esto requiere una formación especializada y una cantidad de tiempo prohibitivamente grande. Afortunadamente, los recientes avances en el aprendizaje automático han hecho posible identificar de manera automática los cantos de aves para especies comunes con una gran cantidad de datos de entrenamiento. Sin embargo, sigue siendo un desafío desarrollar tales herramientas para especies raras y en peligro de extinción, como las de Hawaii.\n",
    "\n",
    "Esta competición busca nuevas ideas para la identificación especies de aves por su sonido. Específicamente, se busca un modelo que pueda procesar datos de audio continuos y luego reconocer acústicamente la especie. De esta manera, se ayudará al avance de la ciencia de la bioacústica y se apoyará la investigación en curso para proteger a las aves hawaianas en peligro de extinción. Gracias a las nuevas propuestas de los concursantes, será más fácil para los investigadores y profesionales de la conservación estudiar con precisión las tendencias de la población. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Primeros pasos\n",
    "\n",
    "### 1. Descarga de los datos en la máquina local\n",
    "\n",
    "Para la descarga de los datos es necesario darse de alta como usuario en [Kaggle](www.kaggle.com). Una vez creado el usuario, se procederá a crear la API necesaria para poder descargar los datos de la competición. Todos los pasos necesarios para su creación y posterior descarga están descritos en la [kaggle API](https://github.com/Kaggle/kaggle-api) de Github. \n",
    "\n",
    "Los datos se descargarán en local directamente desde la terminal de Miniconda mediante ```kaggle competitions download -c birdclef-2022```.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2. EDA: lista de tareas\n",
    "\n",
    "A continuación se enlista las principales tareas a desarrollar para llevar a cabo el Análisis exploratorio de los datos. \n",
    "\n",
    "- [x] Lectura de los archivos desde un documento comprimido .zip\n",
    "- [x] Exploración de los archivos de lectura (e.g., csv)\n",
    "- [ ] Análisis sonoro.\n",
    "- [ ] Elaboración del modelo\n",
    "\n",
    "### 3. Análisis sonoro: parametrización de los archivos de audio\n",
    "***En desarrollo***\n",
    "\n",
    "### 4. Machine Learning\n",
    "***En desarrollo***"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Environment project\n",
    "\n",
    "Para el proyecto, creamos un *environment* que tiene que cumplir los requiesitos descritos en el archivo **requirements.txt**"
   ]
  }
 ],
 "metadata": {
  "interpreter": {
   "hash": "6893aab362e17bc799f78d7c51aedc9fb9447b2289b481b602b16b988e09f948"
  },
  "kernelspec": {
   "display_name": "Python 3.10.4 ('BirdCLEF')",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.4"
  },
  "orig_nbformat": 4
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
