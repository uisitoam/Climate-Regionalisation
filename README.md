# Climate-Regionalization

This project is based on the use of machine learning models to improve the statistical regionalization in areas of complex topography. This technique aims to enhance the relatively poor performance of global climate models in rugged terrains or coastal areas, which do not account for local characteristics. Dynamic simulations executed by The University of La Laguna's Earth and Atmosphere Observation Group will be used for this project.

Convolutional neural networks will be used in order to take into account the non-linear behavior showed in precipitation description; this networks also yield positive outcomes for describing temperature. Following this approach, two similar models will be developed to describe the mentioned variables based on synoptic condictions.

## Script

The following packages are needed to execute the script: `numpy`, `matplotlib`, `tensorflow`, `scipy`, `netCDF4` and `datetime`. The program is divided into five .py scripts:
- `reader.py`: contains the required functions to extract the usefull information from the files where the data under study is given.
- `datafunctions.py`: contains the required functions to manage conveniently the extracted information, then feed to the neural models afterward. It also contains functions to obtain some metrics used to validate the model, along with a function to compute the rainfall amount based on the paramaters of a gamma-Bernouilli distribution given by the precipitation net. 
- `nets.py`: contains the CNN models of temperature and precipitation, along with the loss functions used for each one of them. 
- `trainfunctions.py`: contains the functions related to training the models and making predictions with them, along with the obtention of some metrics used to validate the models.
- `plots.py`: contains the functions used to make the different plots showing the results of the model. 
- `main.py`: this is the script where all functions are called and used to obtain the desired results. 

## Results

Only learning curve of the models and a boxplot of certain metrics. 
