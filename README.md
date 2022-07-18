# FinalPoject
Proyecto final ID Bootcamp Data Science and Machine Learning

## Kaggle competition: BridCLEF 2022

**Objetivo**: identificar cantos de aves mediante grabaciones en campo abierto

**Descripción**
(Resumen de la descripción de [Kaggle](https://www.kaggle.com/competitions/birdclef-2022/overview))

Hawaii ha perdido el 68 % de sus especies de aves alterando otros procesos ecológicos fundamentales (e.g., la alteración de las cadenas de alimento). Los investigadores suelen utilizar métodos de captura directa para la monitorización de poblaciones salvajes, i.e., la captura, marcajae y liberación de individuos. Estas herramientas de seguimiento permiten comprender cómo una población reaccionan a los cambios en el medio ambiente. Desafortunadamente, muchas de las aves de este archipiélago están aisladas en hábitats elevados y de difícil acceso. Estas condiciones dificultan el monitoreo físico, provocando que los científicos recurran a las grabaciones de sonido, también conocido como bioseguimiento o monitoreo bioacústico. Este enfoque podría proporcionar una estrategia pasiva, de bajo costo y rentable para estudiar poblaciones de aves en peligro de extinción.
Los métodos actuales para procesar grandes conjuntos de datos bioacústicos implican la anotación manual de cada grabación. Esto requiere una formación especializada y una cantidad de tiempo prohibitivamente grande. Afortunadamente, los recientes avances en el aprendizaje automático han hecho posible identificar de manera automática los cantos de aves para especies comunes con una gran cantidad de datos de entrenamiento. Sin embargo, sigue siendo un desafío desarrollar tales herramientas para especies raras y en peligro de extinción, como las de Hawaii.
Esta competición busca nuevas ideas para la identificación especies de aves por su sonido. Específicamente, se busca un modelo que pueda procesar datos de audio continuos y luego reconocer acústicamente la especie. De esta manera, se ayudará al avance de la ciencia de la bioacústica y se apoyará la investigación en curso para proteger a las aves hawaianas en peligro de extinción. Gracias a las nuevas propuestas de los concursantes, será más fácil para los investigadores y profesionales de la conservación estudiar con precisión las tendencias de la población.

## Primeros pasos

### 1. Descarga de los datos en la máquina local
Para la descarga de los datos es necesario darse de alta como usuario en [Kaggle](www.kaggle.com). Una vez creado el usuario, se procederá a crear la API necesaria para poder descargar los datos de la competición. Todos los pasos necesarios para su creación y posterior descarga están descritos en la [kaggle API](https://github.com/Kaggle/kaggle-api) de Github. 

Los datos se descargarán en Google Colab mediante un comando mágico y junto al siguiente código: ```kaggle competitions download -c birdclef-2022```. 

### 2. EDA: lista de tareas
A continuación se enlista las principales tareas a desarrollar para llevar a cabo el análisis exploratorio de los datos.
- [x] Lectura de los archivos desde un documento comprimido .zip
- [x] Exploración de los archivos de lectura (e.g., csv)
      - [x] Prospección de la base de datos taxonómica.
      - [x] Prospección de los metadatos.
      - [x] Prospección de las grabaciones disponibles.
- [x] Filtrado de las datos disponibles.
- [x] Selección final de la base de datos.

### 3. Análisis sonoro: parametrización de los archivos de audio
En este apartado, desarrollaremos las funciones necesarias para:
- Leer archivos de audio
- Visualizaciones mediante oscilogramas y espectrogramas
- Limpieza de los sonidos de fondo
La limpieza de los sonidos de fondo es uno de los grandes retos en los análisis de audio. Para la correcta identificación, eliminaremos los que se consideran como ruido estacionario (e.g., el viento).

### 4. Machine Learning
Como modelo de clasificación se a escogido las Redes Neuronales Recurrentes (RNN). Esta metodología es muy utilizada en la clasificación de imágenes. Por ello, los audios se tranformarán en imágenes que representan tres dimensiones del sonido (el tiempo, la amplitud y la frecuencia).

## Environment project
Para el proyecto, creamos un *environment* que tiene que cumplir los requiesitos descritos en el archivo **requirements.txt**"
