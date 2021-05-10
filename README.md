# CorridorTracking
En este proyecto se realiza una aplicación para la detección de suelo en pasillos, así como la obtención de una dirección de movimiento para poder seguir el mismo. Con este fin, gran parte del procesamiento consiste en el uso de la **segmentación semántica** mediante la librería [PixelLib](https://pixellib.readthedocs.io/en/latest/), que permite extraer la parte de la imagen se corresponde con el suelo. El resto del procesamiento se encarga de obtener la dirección de movimiento a partir de la región extraida.

En el repositorio se pueden encontrar las imágenes y los videos de prueba más explicativos. Por limitaciones de tamaño de archivo, el resto de videos utilizados para medir el rendimiento de la aplicación han sido alojados en [Google Drive](https://drive.google.com/drive/folders/1tfJYAgvulws1j3coYbmIPV-TSKLJEWPO?usp=sharing). Estos videos pueden servir para comprobar posibles problemas en el algoritmo de obtención de la dirección.

Por otro lado, para la segmentación semántica se utiliza el modelo [Ade20k](https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.3/deeplabv3_xception65_ade20k.h5) referenciado en la API de [PixelLib](https://pixellib.readthedocs.io/en/latest/image_ade20k.html). **Es necesario descargar este modelo y colocarlo dentro de la carpeta SemanticSegmentation, junto al programa main.py**.

A continuación, se facilitan los diferentes apartados de este readme:

- [Ejecución](#ejecución)
   + [Dependencias principales](#dependencias-principales)
   + [Lanzamiento](#lanzamiento)
   + [Adición de más imágenes y videos](#adición-de-más-imágenes-y-videos)
 - [Estudio](#estudio)
   + [Funciones principales](#funciones)
     - [1. imageSemanticSegmentation](#1-imagesemanticsegmentation)

## Ejecución
Para la ejecución del programa es necesario descargar y colocar el modelo de segmentación tal y como queda explicado en el apartado anterior, así como instalar las dependencias que se comentan más abajo.

### Dependencias principales
Las siguientes dependencias son inherentes a PixelLib:

* pip (versión >= 19.0)
* tensorflow (última versión, 2.0+)
* imgaug 

La instalación de estas dependencias se realiza con pip, tal y como se detalla en la API de PixelLib. Por otro lado, se adjunta la versión utilizada para otras dependencias del código implementado para la obtención de la dirección del pasillo:

* OpenCV (versión 4.5.1.48)
* numpy (versión 1.19.5)

### Lanzamiento
Una vez instaladas las dependencias indicadas anteriormente, para lanzar el programa basta con llamar al intérprete de Python:

```diff
python3 main.py
```

**Nota:** El código implementado ha sido ideado para Python 3. En caso de usar Python 2, inicialmente sería necesario especificar el uso de UTF-8 para evitar errores de ejecución.

### Adición de más imágenes y videos
Es posible añadir más imágenes y videos con los que poder ejecutar el algoritmo. En el caso de las imágenes, estas tienen que tener formato **.jpg** y ser añadidas a la carpeta **img**. Para los videos, el formato aceptado es **.mp4** y deben ser añadidos a la carpeta **video**. 

## Estudio
### Funciones
En este apartado se pretende dar una idea general del contenido del fichero main.py para acercar al lector los algoritmos utilizados, explicando por encima todas las funciones implementadas:

#### 1. imageSemanticSegmentation
Esta función es llamada al seleccionar la primera opción del menú principal de la aplicación (imágenes), y se encarga de explorar la carpeta img en busca de las imágenes a las que todavía no se ha realizado la segmentación semántica, es decir, no tienen una imagen con terminación "\_seg" asociada. Si existen imágenes sin segmentar, se utiliza el modelo Ade20k para la segmentación de todas estas imágenes y se miden los tiempos, que al finalizar el proceso son mostrados por consola. 

#### 2. videoSemanticSegmentation
Esta función es llamada al seleccionar la segunda opción del menú principal de la aplicación (videos). Si el video especificado (en el caso de existir) no tiene asociado un video con la terminación "\_seg", se realiza su segmentación semántica haciendo uso del modelo Ade20k. Al igual que con la función anterior, al finalizar el proceso se muestran los tiempos por consola.

