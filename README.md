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

## Estudio de la implementación
### Funciones
En este apartado se pretende dar una idea general del contenido del fichero main.py para acercar al lector los algoritmos utilizados, explicando por encima todas las funciones implementadas:

#### 1. imageSemanticSegmentation
Esta función es llamada al seleccionar la primera opción del menú principal de la aplicación (imágenes), y se encarga de explorar la carpeta img en busca de las imágenes a las que todavía no se ha realizado la segmentación semántica, es decir, no tienen una imagen con terminación "\_seg" asociada. Si existen imágenes sin segmentar, se utiliza el modelo Ade20k para la segmentación de todas estas imágenes y se miden los tiempos, que al finalizar el proceso son mostrados por consola. 

#### 2. videoSemanticSegmentation
Esta función es llamada al seleccionar la segunda opción del menú principal de la aplicación (videos). Si el video especificado (en el caso de existir) no tiene asociado un video con la terminación "\_seg", se realiza su segmentación semántica haciendo uso del modelo Ade20k. Al igual que con la función anterior, al finalizar el proceso se muestran los tiempos por consola.

#### 3. rescale
Esta función se utiliza principalmente para evitar que las imágenes mostradas por pantalla tengan una resolución mayor que la resolución de la misma, de tal forma que no se puedan visualizar correctamente los resultados del programa. Para ello, se fija un tamaño máximo de 600 píxeles. La dimensión (alto o ancho) que tenga mayor tamaño y supere este tamaño máximo queda limitada al mismo, y la dimensión restante queda reescalada en la misma proporción.

#### 4. extractFloor
Esta función permite la extracción de las zonas clasificadas como _suelo_ por el modelo Ade20k. Para ello, se tiene en cuenta el hecho de que el modelo asocia a las zonas de tipo _suelo_ el color RGB \[50, 50, 80\]. También se tiene en cuenta el hecho de que las áreas segmentadas no tienen los bordes especialmente definidos, algo que se intenta solucionar proporcionando un rango pequeño de colores en torno al color objetivo.

#### 5. limpiaMemoria
Esta función permite vaciar los arrays de memoria utilizados para el cálculo de la dirección del pasillo, empleados en la función floorAndContours que queda explicada más abajo.

#### 6. midPoints
Esta función implementa la extracción de los dos puntos clave para la obtención de la dirección del pasillo, y supone prácticamente la mitad del código implementado. Para dicha extracción se utiliza una aproximación poligonal a la zona extraída con la función extractFloor. La función queda dividida en varias partes:

##### Búsqueda inicial
Inicialmente se buscan los dos puntos de menor (punto superior) y menor (punto inferior) coordenada Y. 

##### Punto medio del segmento superior
Puesto que el polígono que aproxima la zona extraída no tiene por qué ser necesariamente un triángulo, se añaden una serie de mejoras que intentan contemplar diversos casos. Esta es la primera de las mejoras.

Sea **minY** el punto de menor coordenada Y (punto superior) y **pSig y pAnt** el punto siguiente y anterior respectivamente en la aproximación poligonal (formado por un conjunto de puntos cuyo orden determina los segmentos de la misma), inicialmente se calculan los puntos medios de los **segmentos minY-pAnt y minY-pSig**, así como las pendientes entre estos nuevos puntos medios y el punto minY.


