# Climate-Regionalization

This project is based on the use of machine learning models to improve the statistical regionalization in areas of complex topography. This techniques aims to enhace the relatively poor performance of global climate models in rugged terrains or coastal aereas, which do not allow for local characteristics. Dynamic simulations executed by The University of La Laguna's Earth and Atmosphere Observation Group will be used for this project.

Convolutional neural networks will be used in order to take into account the non-linear behavior showed in precipitation description; this networks also yield positive outcomes for describing temperature. Following this approach, two similar models will be developed to describe the mentioned variables based on synoptic condictions.

## Código

Para la ejecución del código se requieren los siguientes módulos: `numpy`, `matplotlib`, `tensorflow`, `scipy`, `netCDF4` y `datetime`. El código se divide en varios archivos:
- `reader.py`: contiene las funciones necesarias para extraer toda la información útil de los archivos donde se presentan los datos sujetos a estudio.
- `datafunctions.py`: contiene las funciones necesarias para tratar la información extraída de forma conveniente para posteriormente entrenar los modelos neuronales.
- `nets.py`: contiene los modelos de temperatura y precipitaciones, así como las funciones de pérdida usadas en cada uno.
- `trainfunctions.py`: contiene las funciones relativas al entrenamiento y predicción de los modelos, así como el cálculo de las métricas usadas para la validación de los propios modelos.

## Resultados 

Ya se irá viendo 
