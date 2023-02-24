# Reconocimiento de lenguaje de señas aislado

<img src="img/pipeline overview.png" alt="Estructura general del proceso" style="center"/>

En este repositorio se encuentra el código fuente de nuestro trabajo de tesis de grado en la Universidad Simón Bolívar. Para este proyecto se desarrolló un sistema de reconocimiento de lenguaje de señas aislado a nivel de _Gloss_. Dado un vídeo con una ejecución de una seña, se retorna la clase predicha para esta seña. El proceso consiste en extraer anotaciones esqueléticas para cada fotograma el video usando estimación de pose con [Mediapipe](https://mediapipe.dev), y luego representar la secuencia de grafos esqueléticos de dos formas distintas:

- Se define el **mapa de trayectoria**, una imagen que representa como una nube de puntos a los puntos de articulación a través del tiempo, marcando en colores cada parte de cuerpo. Las partes del cuerpo consideradas son: mano izquierda, mano derecha, y postura, que corresponde a los brazos, el torso, los hombros y algunos puntos en el rostro. 

<img src="img/pipeline_construcción_de_mapas_de_trayectoria (1).png" alt="Creación de un mapa de trayectoria" style="display: block; margin: 0 auto"/>

- Se normalizan las posiciones de los puntos de articulación y se concatenan por frame creando un único vector que contiene a todos los demás. De esta forma, una seña consiste en una secuencia de estos vectores, que se concatenan de nuevo para formar una **matriz de características**.

<img src="img/concatenación de vectores aplanados (1).png" alt="Creación de un mapa de trayectoria" style="display: block; margin: 0 auto"/>

Estas dos nuevas representaciones se pasan a un modelo de dos canales que procesa el mapa de trayectoria con un módulo CNN para obtener características espaciales, y la matriz de características con un módulo LSTM para obtener características temporales. Finalmente, el resultado de ambos módulos se procesa y se agrega usando la operación de promedio término a término.
## Entrenamiento
El entrenamiento se realizó con las bases de datos: [MS-ASL](https://www.microsoft.com/en-us/download/confirmation.aspx?id=100121), [LSA64](http://facundoq.github.io/datasets/lsa64/). Obteniendo los siguientes resultados:

El entrenamiento se realizó en una tarjeta gráfica Nvidia GeForce RTX 2060 de 6GB de RAM. 

## Instalación
La aplicación usa [Poetry](https://python-poetry.org) como manejador de paquetes, para instalarla es suficiente con clonar este repositorio, cambiar de directorio a la carpeta `usb_slr` y ejecutar el comando `poetry install`. Una vez termine el proceso de instalación, se habrá añadido un programa de línea de comandos `slr` que sirve para realizar operaciones comunes. 

## Uso
La aplicación provista en este repositorio incluye una herramienta de línea de comandos que sirve como interfaz para operaciones comunes sobre las bases de datos, como cargar datos, procesarlos, y ejecutar un entrenamiento.
