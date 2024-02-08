# Climate-Regionalization

Aplicación de sistemas de aprendizaje automático para mejorar la regionalización estadística en áreas de orografía compleja. Esta técnica permite mejorar los pobres resultados que dan los modelos climáticos globales en terrenos abruptos o zonas costeras al no considerar características propias. Para esto, se aprovecharán diversas simulaciones dinámicas realizadas por el Grupo de Observación de la Tierra y la Atmósfera de la Universidad de La Laguna para la región de Canarias.

Se tratará con redes convolucionales para considerar las no linealidades presentes en la descripción de las precipitaciones; estas también muestran buenos resultados para la descripción de temperaturas. Siguiendo esta línea, se realizarán dos modelos similares capaces de describir las variables mencionadas en función de las condiciones sinópticas. 

## Código

El código se divide en varios archivos:
- `reader.py`: contiene las funciones necesarias para extraer toda la información útil de los archivos donde se presentan los datos sujetos a estudio.
- `datafunctions.py`: contiene las funciones necesarias para tratar la información extraída de forma conveniente para posteriormente entrenar los modelos neuronales.
- `nets.py`: contiene los modelos de temperatura y precipitaciones, así como las funciones de pérdida usadas en cada uno.
- `trainfunctions.py`: contiene las funciones relativas al entrenamiento y predicción de los modelos, así como el cálculo de las métricas usadas para la validación de los propios modelos.

## Resultados 

Ya se irá viendo 
