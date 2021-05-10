# CorridorTracking
En este proyecto se realiza una aplicación para la detección de suelo en pasillos, así como la obtención de una dirección de movimiento para poder seguir el mismo. Con este fin, gran parte del procesamiento consiste en el uso de la **segmentación semántica** mediante la librería [PixelLib](https://pixellib.readthedocs.io/en/latest/), que permite extraer la parte de la imagen se corresponde con el suelo. El resto del procesamiento se encarga de obtener la dirección de movimiento a partir de la región extraida.

En el repositorio se pueden encontrar las imágenes y los videos de prueba más explicativos. Por limitaciones de tamaño de archivo, el resto de videos utilizados para medir el rendimiento de la aplicación han sido alojados en [Google Drive](https://drive.google.com/drive/folders/1tfJYAgvulws1j3coYbmIPV-TSKLJEWPO?usp=sharing). Estos videos pueden servir para comprobar posibles problemas en el algoritmo de obtención de la dirección.

Por otro lado, para la segmentación semántica se utiliza el modelo [Ade20k](https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.3/deeplabv3_xception65_ade20k.h5) referenciado en la API de [PixelLib](https://pixellib.readthedocs.io/en/latest/image_ade20k.html). **Es necesario descargar este modelo y colocarlo dentro de la carpeta SemanticSegmentation, junto al programa main.py**.

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

### Funciones principales

-- TODO: Comentar todas las funciones
