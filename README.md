# Anomaly_detection
Detección de actividades criminales en cámaras de seguridad

Este es un código complementario al código [STEAD](https://github.com/agao8/STEAD), por lo que **se debe clonar aquel repositorio, descargar sus datos y agregar los códigos de este repositorio**.

## Descripciones archivos

- En la carpeta `MyVideos` van los videos propios (agenos al datset original) para probar el código. Esta está subcompuesta de dos carpetas: `Anomaly` (videos con anomalías/crímenes) y `Normal` (videos normales, sin anomalias). 

- En la `carpeta X3D_Features` se guardan los `.npy` de las características extraídas. Esta está subcompuesta de dos carpetas: `Anomaly` (características de videos con anomalías/crímenes) y `Normal` (características de videos normales, sin anomalias). 

- `check_npy_shapes.py` chequea los tamaños de las caractpsiticas extraídas (depende del modelo). 

- `diagnose_videos_and_generate_list.py` chequea si los videos que se usarán son válidos o no (por problemas en los videos o su duración), y genera una lista en `my_test.txt`, la cual contiene una lista con las direcciones a los archivos de características. 

- `extract_my_videos_final.py` genera las características de los videos de `MyVideos`. Genera la carpeta X3D_Features y la lista de texto `my_test.txt`. **IMPORTANTE: este código aún no funciona bien y no he logrado encontrar el problema.** Este código está basado en el códgio `feat_extractor.py` del repo original. Recomiendo arreglar este código en base a ese.

- `predict_my_videos.py` predice la **asertividad promedio** (ie: calcula las asertividades por extracto de video, y promedia estos resultado, haciendo **una** clasificación por video más que por momento de video) de si un video es Anomaly o Normal. Este códgo funciona bien, ya que fue probado con características del dataset original.

## Ambientes virtuales

Es importante que se generen ambientes virtuales para hacer funcionar el código. En este caso se deben usar dos ambientes distintos: uno para extraer características (al correr `extract_my_videos_final.py` o su versión arreglada) y otro para la clasificación (al correr `predict_my_videos.py`). Esto se da por problemas de versiones a la fecha.

- `requirements_extract.txt`: requerimientos del ambiente para correr la extracción de características.
-  `requirements_model.txt`: requerimientos del ambiente para correr la clasificación de los videos.
