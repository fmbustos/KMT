# KMT: Kauel Motion Tracking

<!-- [![Maintenance](https://img.shields.io/badge/maintained-yes-brightgreen.svg)](https://gitHub.com/rsalmei/alive-progress/graphs/commit-activity)
[![PyPI status](https://img.shields.io/pypi/status/alive-progress.svg)](https://pypi.python.org/pypi/alive-progress/)
[![PyPI version](https://img.shields.io/pypi/v/alive-progress.svg)](https://pypi.python.org/pypi/alive-progress/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/alive-progress.svg)](https://pypi.python.org/pypi/alive-progress/)
[![Downloads](https://static.pepy.tech/personalized-badge/alive-progress?period=week&units=international_system&left_color=grey&right_color=orange&left_text=downloads/week)](https://pepy.tech/project/alive-progress)
[![Downloads](https://static.pepy.tech/personalized-badge/alive-progress?period=month&units=international_system&left_color=grey&right_color=orange&left_text=/month)](https://pepy.tech/project/alive-progress)
[![Downloads](https://static.pepy.tech/personalized-badge/alive-progress?period=total&units=international_system&left_color=grey&right_color=orange&left_text=total)](https://pepy.tech/project/alive-progress) -->

Programa de rastreo de movimiento para deportistas desarrollado por Kauel, `KMT` o Kauel Motion Tracking utiliza mediapipe para calcular los distintos ángulos y velocidades del cuerpo humano desde un video para su análisis.
---
## Tabla de Contenidos

<!-- TOC -->
* [KMT](#KMT:-Kauel-Motion-Tracking)
  * [Table of contents](#tabla-de-contenidos)
  * [Usando KMT](#Usando-KMT)
      * [Instalarlo](#Instalarlo)
      * [Usarlo](#Usarlo)
  * [Funciones](#Funciones)
    * [kvideo_analysis](#kvideo_analysis)
    * [ksave_analysis](#ksave_analysis)
    * [kload_analysis](#kload_analysis)
    * [kplot_landmarks_world](#kplot_landmarks_world)
    * [kplot_landmarks](#kplot_landmarks)
    * [kplot_angles](#kplot_angles)
    
<!-- TOC -->
---

## Usando `KMT`

### Instalarlo

Primero, hay que tener instalado python, la versión no debe ser menor a 3.6 ni mayor a 3.10, preferiblemente la última.
Luego, bajar o clonar el repositorio desde github.
Abrir la consola y cambiar de directorio a la carpeta de 'KMT', desde ahí instalar los requerimientos usando pip.
Sólo es necesario instalar requiriments.txt para usar el módulo 'KMT0', que contiene las funciones de análisis de Motion Tracking.
Sin embargo se recomienda instalar requiriments2.txt, para utilizar el módulo 'KMT', que contiene las mismas funciones de ánalisis, pero muestra en la consola el progreso de las distintas funciones.

```sh
❯ cd c:\\path\\KMT
❯ pip install -r requirements.txt
❯ pip install -r requirements2.txt
```
### Usarlo

En un archivo python importar el módulo 'KMT' o 'KMT0', preferiblemente el primero. Es importante que si se están importando todo el contenido de 'KMT', no importar al mismo tiempo todo el contenido de 'KMT0', ya que las funciones tienen el mismo nombre.

```python
from KMT0 import *
#Nunca al mismo tiempo que:
from KMT import *

```
También se puede importar cada módulo por separado como:
```python
import KMT as kmt
import KMT0 as kmt0

```

O sólo la función que se necesita:

```python
from KMT import kload_analysis


```
---
## Funciones

Las funciones principales del módulo 'KMT':

### 'kvideo_analysis()'

Analiza un video y devuelve una tupla con tres listas, una con las coordenadas reales aproximadas de 33 puntos claves del cuerpo, la segunda con las coordenadas en la imagen de los mismos puntos claves y una lista de cada imagen del video en la forma de un arreglo de numpy (numpy array)

Entrada:
- video_file: Ruta al video

Salida:
- video_landmarks_world, video_landmarks_image, image_list
```python
from KMT import *

video_file = 'C:\\path\\to\\video.mp4'

#lws: video landmarks world, lws: video landmarks image, ims: video images
lws, lis, ims = kvideo_analysis(video_file)
```
### 'ksave_analysis()'

Guarda la información de 'kvideo_analysis()' en la carpeta asignada, los landmarks world como lws.json, landmarks image como lis.json y las imagenes como un video llamado ims.avi. Devuelve la misma tupla con 3 listas que se le entregó.

Entrada:

- video_data: Tupla con la información del análisis en el mismo orden que retorna 'kvideo_analysis()'.
- folder_name: Default = analysis, Nombre de la carpeta, si no existe esta función la creará.
- data_name: Opcional, para guardar informacion diferente pero relacionada en la misma carpeta, todos los datos guardados tendrán este nombre más un indicador del tipo de información.
- landmarks_world_name: Opcional, para guardar landmarks world diferentes pero relacionados en la misma carpeta, ignorará el nombre de data_name.
- landmarks_image_name: Opcional, para guardar landmarks image diferentes pero relacionados en la misma carpeta, ignorará el nombre de data_name.
- video_name: Opcional, para guardar imagenes diferentes pero relacionados en la misma carpeta, ignorará el nombre de data_name.

Salida:

- video_landmarks_world, video_landmarks_image, image_list
```python
from KMT import *

video_file = 'C:\\path\\to\\video.mp4'

#Consiguiendo la información
video_data = kvideo_analysis(video_file)

#Guardando en una nueva carpeta llamada 'analysis'
ksave_analysis(video_data)

#En caso de tener la información separada hay que entregar la tupla a la función.
lws, lis, ims = kvideo_analysis(video_file)

ksave_analysis((lws, lis, ims))

#Para guardar en una carpeta con distinto nombre, que se va a guardar en la misma carpeta donde esté el programa. También se le puede dar la ruta a la nueva carpeta si es que se quiere guardar en un directorio distinto.

new_folder = 'new_analysis_folder'
path_to_new_folder = 'C:\\path\\to\\directory\\new_video_analysis'

ksave_analysis(video_data, folder_name = new_folder)

ksave_analysis(video_data, folder_name = path_to_new_folder)

#Guardando información relacionada en la misma carpeta.
video_file1 = 'C:\\path\\to\\video1.mp4'
video_file2 = 'C:\\path\\to\\video2.mp4'
video_data1 = kvideo_analysis(video_file1)
video_data2 = kvideo_analysis(video_file2)

ksave_analysis(video_data1, data_name = 'data1')
ksave_analysis(video_data2, data_name = 'data2')
#En la nueva carpeta se verán los archivos data1_lws.json, data1_lis.json y data1_ims.avi junto con data2_lws.json, data2_lis.json y data2_ims.avi 
```

### 'kload_analysis()': 
Carga la información guardada en la carpeta creada por 'ksave_analysis()'

Entrada:

- folder_name: Default = analysis, Nombre de la carpeta donde está guardada la información.
- data_name: Opcional, cuando hay información diferente pero relacionada en la misma carpeta, carga todos los datos guardados con este nombre.
- landmarks_world_name: Opcional, cuando hay landmarks world diferentes pero relacionados en la misma carpeta, cargará los datos con este nombre.
- landmarks_image_name: Opcional, cuando hay landmarks images diferentes pero relacionados en la misma carpeta, cargará los datos con este nombre.
- video_name:Opcional, cuando hay videos diferentes pero relacionados en la misma carpeta, cargará los datos con este nombre.

Salida:

- video_landmarks_world, video_landmarks_image, image_list
```python
from KMT import *

video_file = 'C:\\path\\to\\video.mp4'
new_folder = 'new_analysis_folder'
path_to_new_folder = 'C:\\path\\to\\directory\\new_video_analysis'

#Cargando la información guardada en la carpeta default.
lws, lis, ims = kload_analysis('analysis')

#Cargando desde en una carpeta en el mismo directorio que el programa.
lws, lis, ims = kload_analysis(new_folder)

#Cargando usando la ruta absoluta a la carpeta.
lws, lis, ims = kload_analysis(path_to_folder)

#Cargando desde una carpeta con múltiples datos.
data1 = kload_analysis(path_to_folder, data_name = 'data1')
data2 = kload_analysis(path_to_folder, data_name = 'data2')
```

### 'kplot_landmarks_world()'
Abre una ventana donde grafica los landmarks en una grilla 3d, y si se le entrega lo muestra al lado del video.
    
Entrada:

- video_landmarks_world: Video landmarks entregadas por 'kvideo_analysis()'
- video_images: Opcional, lista de imagenes entregadas por 'kvideo_analysis()'
- plot_body_axis: Opcional, lista con nombres de distintas partes del cuerpo para mostrar ejes de referencia relativos a esa parte del cuerpo. Las opciones válidas son: 'central', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'
- plot_angles: Opcional, lista con nombres de distintas partes del cuerpo para mostrar los distintos ángulos que produce con otras partes del cuerpo o con ejes de referencia. Las opciones válidas son: 'left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow', 'left_hip', 'left_knee', 'right_hip',  'right_knee'
- save: Booleano opcional, 'default = True'. Si es True guardará un video con los graficos animados como un .gif.
- filename: 'Default = angles_plot.gif', si es que save = True, entonces guardará el la animación con este nombre. Se guardará en el mismo directorio donde esté el .py del programa a menos a que se le de la ruta absoluta. Recordar escribir .gif al final del nombre.
- fps: Default = 30, si es que save = True, entonces la animación guardada tendrá este frame rate.

Salida:
- True si es que termine el proceso sin problemas.
```python
from KMT import *

video_file = 'C:\\path\\to\\video.mp4'

#Datos del análisis de un video.
lws, lis, ims = kvideo_analysis(video_file)

#Graficando una animación de los landmarks world.
kplot_landmarks_world(lws)

#Graficando landmarks world junto con el video.
kplot_landmarks_world(lws, ims)

#Graficando con los distintos ángulos y ejes de referencia.
axis = ['central', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
angles = ['left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow', 'left_hip', 'left_knee', 'right_hip',  'right_knee']

kplot_landmarks_world(lws, ims, plot_body_axis = axis, plot_angles = angles)

#Graficando el lado izquierdo y guardando la animación.
filename_path = 'C:\\path\\to\\folder\\landmarks_animation.gif'
fps = 10

axis = ['left_shoulder', 'left_hip']
angles = ['left_shoulder', 'left_elbow', 'left_hip', 'left_knee']

kplot_landmarks_world(lws, ims, plot_body_axis = axis, plot_angles = angles, save = True, filename = filename_path, fps = fps)
```

### 'kplot_landmarks()'
Abre una ventana donde grafica múltiples landmarks en 3D.
    
Entrada:

- *landmarks_set: Múltiples video landmarks entregadas por 'kvideo_analysis()', separadas por comas.
- plot_body_axis: Opcional, lista con nombres de distintas partes del cuerpo para mostrar ejes de referencia relativos a esa parte del cuerpo. Las opciones válidas son: 'central', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'
- plot_angles: Opcional, lista con nombres de distintas partes del cuerpo para mostrar los distintos ángulos que produce con otras partes del cuerpo o con ejes de referencia. Las opciones válidas son: 'left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow', 'left_hip', 'left_knee', 'right_hip',  'right_knee'
- save: Booleano opcional, 'default = True'. Si es True guardará un video con los graficos animados como un .gif.
- filename: 'Default = angles_plot.gif', si es que save = True, entonces guardará el la animación con este nombre. Se guardará en el mismo directorio donde esté el .py del programa a menos a que se le de la ruta absoluta. Recordar escribir .gif al final del nombre.
- fps: Default = 30, si es que save = True, entonces la animación guardada tendrá este frame rate.

Salida:

- True si es que termine el proceso sin problemas.
```python
from KMT import *

video_file = 'C:\\path\\to\\video.mp4'
data_folder = 'C:\\path\\to\\analysis_folder'
data1_name = 'data1'
data2_name = 'data2'

#Datos del análisis de un video.
lws, lis, ims = kvideo_analysis(video_file)
data1_lws, data1_lis, data1_ims = kload_analysis(data_folder, data_name = data1_name)
data2_lws, data2_lis, data2_ims = kload_analysis(data_folder, data_name = data2_name)

#Graficando una animación de los landmarks world.
kplot_landmarks_world(lws)

#Graficando landmarks world junto con el video.
kplot_landmarks_world(lws, data1_lws, data2_lws)

#Graficando con los distintos ángulos y ejes de referencia.
axis = ['central', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
angles = ['left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow', 'left_hip', 'left_knee', 'right_hip',  'right_knee']

kplot_landmarks_world(lws, data1_lws, data2_lws, plot_body_axis = axis, plot_angles = angles)

#Graficando el lado izquierdo y guardando la animación.
filename_path = 'C:\\path\\to\\folder\\landmarks_animation.gif'
fps = 10

axis = ['left_shoulder', 'left_hip']
angles = ['left_shoulder', 'left_elbow', 'left_hip', 'left_knee']

kplot_landmarks_world(lws, data1_lws, data2_lws, plot_body_axis = axis, plot_angles = angles, save = True, filename = filename_path, fps = fps)
```

### 'kplot_angles()'
Abre una ventana donde grafica landmarks en una grilla 3D, las imagenes analizadas junto con representaciones de los ángulos que forman distintas partes del cuerpo.

Entrada:

- video_landmarks_world: Video landmarks entregadas por 'kvideo_analysis()'
- video_images: Opcional, lista de imagenes entregadas por 'kvideo_analysis()'
- angles: 'Default = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_hip', 'right_hip', 'left_knee','right_knee']' Lista con los nombres de las partes del cuerpo de las que se quiere graficar sus ángulos.
- plot_body_axis: Opcional, lista con nombres de distintas partes del cuerpo para mostrar ejes de referencia relativos a esa parte del cuerpo. Las opciones válidas son: 'central', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'
- plot_angles: Opcional, lista con nombres de distintas partes del cuerpo para mostrar los distintos ángulos que produce con otras partes del cuerpo o con ejes de referencia en el gráfico 3D. Las opciones válidas son: 'left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow', 'left_hip', 'left_knee', 'right_hip',  'right_knee'
- save: Booleano opcional, 'default = True'. Si es True guardará un video con los graficos animados como un .gif.
- filename: 'Default = angles_plot.gif', si es que save = True, entonces guardará el la animación con este nombre. Se guardará en el mismo directorio donde esté el .py del programa a menos a que se le de la ruta absoluta. Recordar escribir .gif al final del nombre.
- fps: Default = 10, si es que save = True, entonces la animación guardada tendrá este frame rate.

Salida:

- True si es que termine el proceso sin problemas.
```python
from KMT import *

video_file = 'C:\\path\\to\\video.mp4'
angles = ['left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow', 'left_hip', 'left_knee', 'right_hip',  'right_knee']
axis = ['central', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']

#Datos del análisis de un video.
lws, lis, ims = kvideo_analysis(video_file)

#Graficando una animación de todos los ángulos.
kplot_angles(lws, ims)

#Graficando una animación de todos los ángulos, junto con dibujando dichos ángulos en el modelo 3D
kplot_angles(lws, ims, plot_body_axis = axis, plot_angles = angles)

#Graficando el lado izquierdo y guardando la animación.
filename_path = 'C:\\path\\to\\folder\\angles_animation.gif'
fps = 10

axis = ['left_shoulder', 'left_hip']
angles = ['left_shoulder', 'left_elbow', 'left_hip', 'left_knee']

kplot_angles(lws, ims, plot_body_axis = axis, angles = angles, plot_angles = angles, save = True, filename = filename_path, fps = fps)
```
---