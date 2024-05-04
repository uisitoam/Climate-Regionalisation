import os
import numpy as np
from nets import temperatureModel, precipModel
from datafunctions import coeficientes, computeRainfall
from plots import learningCurve





def entrenamiento(model, loss, data, params):
    """
    This function supports training for both temperature and precipitation models with
    different loss functions. It iterates through training sessions based on the 'repeats'
    parameter, accumulates training loss and validation loss, and saves the model and
    training statistics. Additionally, it generates learning curves for the training
    process.

    Parameters
    ----------s
    model : function
        The model function to be trained. This can be either the temperatureModel or
        the precipModel, which are designed for temperature and precipitation prediction,
        respectively.
    loss : function
        The loss function to be used for training the model. This function should be
        appropriate for the type of prediction being made (i.e., a Gaussian loss for
        temperature predictions and a Bernoulli-Gamma loss for precipitation predictions).
    data : list of np.ndarray
        A list containing two numpy arrays: the first array contains the training samples,
        and the second array contains the corresponding labels for training.
    params : list
        A list containing three training parameters: the number of epochs for training,
        the batch size to be used, and the number of times the training should be repeated
        to average out the training and validation losses.

    Returns
    -------
    None
        This function does not return any value. It saves the trained models and their
        training statistics to disk and generates learning curve plots.

    Raises
    ------
    AssertionError
        Raised if the input parameters do not meet the expected format or type: 
        - model must be a function.
        - loss must be a function.
        - data must be a list of two elements.
        - The first element of data must be a 4D array.
        - The second element of data must be a 2D array.
        - The first dimension of both arrays in data must match.
        - params must be a list of three integers.

    """

    # check input parameters
    assert callable(model), "model must be a function."
    assert callable(loss), "loss must be a function."
    
    assert isinstance(data, list) and len(data) == 2, "data must be a list of two elements."
    assert isinstance(data[0], np.ndarray) and data[0].ndim == 4, "The first element of data must be a 4D array."
    assert isinstance(data[1], np.ndarray) and data[1].ndim == 2, "The second element of data must be a 2D array."
    assert data[0].shape[0] == data[1].shape[0], "The first dimension of both arrays in data must match."
    assert isinstance(params, list) and len(params) == 3 and all(isinstance(p, int) for p in params), "params must be a list of three integers."


    train_loss, train_val = 0, 0 # accumulate training and validation losses
    samples, labels = data

    # training process for temperature model
    if model == temperatureModel:
        for i in range(params[2]):
            modelo, historial = model(samples, labels, loss, params[:2])

            try:
                if not os.path.exists('Resultados/Temperatura/modelos'):
                    os.makedirs('Resultados/Temperatura/modelos') # if it does not exist, create it

            except Exception as e:
                raise Exception(f"Error al crear la carpeta: {e}")
            
            modelo.save(f'./Resultados/Temperatura/modelos/temp({i}).keras')
            
            # accumulate training and validation losses
            train_loss += np.array(historial.history['loss'])
            train_val += np.array(historial.history['val_loss'])
    
    # training process for precipitation model
    elif model == precipModel:
        for i in range(params[2]):
            modelo, historial, trainProb = model(samples, labels, loss, params[:2])

            try:
                if not os.path.exists('Resultados/Precipitacion/modelos'):
                    os.makedirs('Resultados/Precipitacion/modelos') # if it does not exist, create it

            except Exception as e:
                raise Exception(f"Error al crear la carpeta: {e}")
            
            modelo.save(f'./Resultados/Precipitacion/modelos/precip({i}).keras')

            # save training probabilities and accumulate losses
            try:
                if not os.path.exists('Resultados/Precipitacion/datos'):
                    os.makedirs('Resultados/Precipitacion/datos') # if it does not exist, create it

            except Exception as e:
                raise Exception(f"Error al crear la carpeta: {e}")
            
            np.save(f'./Resultados/Precipitacion/datos/precipTrainProb({i})', trainProb)   
            train_loss += np.array(historial.history['loss'])
            train_val += np.array(historial.history['val_loss'])


    # calculate average training and validation losses
    training_loss = train_loss/params[2]
    training_val = train_val/params[2]

   
    # generate and save learning curves
    if model == temperatureModel:
        learningCurve([training_loss, training_val], 
                         ['Resultados/Temperatura/plots', f'{params[0]}epochs_{params[1]}bs_Learning_curve(temp).pdf'])
    
    elif model == precipModel:
        learningCurve([training_loss, training_val], 
                         ['Resultados/Precipitacion/plots', f'{params[0]}epochs_{params[1]}bs_Learning_curve(precip).pdf'])
    
    return





def predicciones(models, data, store, times, mask, trainProb=None, stochastic = False):
    """
    This function generates predictions for temperature and precipitation using a list of five
    trained models. It computes the average of the predictions and the corresponding metrics
    for the predictions. The function also supports stochastic simulation of precipitation
    predictions based on the training probabilities.

    Parameters
    ----------
    models : list
        A list containing five trained keras models.
    data : list of np.ndarray
        A list containing three numpy arrays: the first array contains the training labels,
        the second array contains the test samples, and the third array contains the
        corresponding labels for test.
    store : dict
        A dictionary containing the maximum and minimum values for precipitation samples.
    times : np.ndarray
        A numpy array containing the time values for the test samples and labels.
    mask : np.ndarray
        A numpy array containing the mask values for the test samples and labels.
    trainProb : np.ndarray, optional
        A numpy array containing the training probabilities for precipitation predictions.
        The default is None.
    stochastic : bool, optional
        A boolean flag to enable stochastic simulation of precipitation predictions.
        The default is False.

    Returns
    -------
    predicciones : np.ndarray
        A numpy array containing the average predictions for temperature or precipitation.
    metricas : np.ndarray
        A numpy array containing the metrics for the predictions.

    Raises
    ------
    AssertionError
        Raised if the input parameters do not meet the expected format or type: 
        - models must be a list of five keras Models.
        - data must be a list of three numpy arrays.
        - The first array in data must contain the training labels.
        - The second array in data must contain the test samples.
        - The third array in data must contain the test labels.
        - The first dimension of the second and third arrays in data must match.
        - store must be a dictionary.
        - times must be a numpy array of strings.
        - The length of times must match the first dimension of the second and third arrays in data.
        - mask must be a 1D numpy array.
        - trainProb must be a numpy array.
        - stochastic must be a boolean.

    """
    
    # check input parameters
    assert isinstance(models, list) and len(models) == 5, "models must be a list of five keras Models."
    assert isinstance(data, list) and len(data) == 3 and all(isinstance(d, np.ndarray) for d in data), "data must be a list of three NumPy arrays."
    assert data[1].shape[0] == data[2].shape[0], "The first dimension of the data[1] and data[2] must match."
    assert isinstance(store, dict), "store must be a dictionary."
    assert isinstance(times, np.ndarray) and times.dtype.type is np.str_, "times must be a NumPy array of strings."
    assert times.size == data[1].shape[0], "The length of times must be equal to the first dimension of data[1] and data[2]."
    assert isinstance(times, np.ndarray) and times.dtype.type is np.str_, "times must be a NumPy array of strings."
    
    
    assert isinstance(mask, np.ndarray) and mask.ndim == 1, "mask must be a 1D NumPy array."
    
    if trainProb is not None:
        assert isinstance(trainProb, list) and all(isinstance(t, np.ndarray) for t in trainProb), "trainProb must be a NumPy array."
    
    assert isinstance(stochastic, bool), "stochastic must be a boolean."


    res, coefs = 0, 0 # accumulate predictions and metrics
    trainLabels, samples, labels = data

    for i, model in enumerate(models): 
        preds = model.predict(samples) 

        if trainProb is not None:
            amountData, prob = computeRainfall(preds, simulate=stochastic)

            porcentaje = np.sum(trainLabels <= 0, axis=0)/np.shape(trainLabels)[0] # (1059,)
            cota = np.diag(np.nanquantile(trainProb[i], porcentaje, axis=0)) # (1059,)
            predictions = np.where(prob < cota, 0, amountData) # (t, 1059)

            coefis = coeficientes(predictions, labels, list(times), 'pr')
            
        else: 
            predictions = np.interp(preds[:, :int(np.shape(preds)[1]/2)], (0, 1), 
                                    (store['labels(t)'][0], store['labels(t)'][1])) # (t, 1050)
            #predictions = predictions[:, mask] # mascara de tierra (t, 1059)

            coefis = coeficientes(predictions, labels, list(times), 'temp')

        res += predictions
        coefs += coefis
    
    # average predictions and metrics
    predicciones = res/len(models)
    metricas = coefs/len(models)

    return predicciones, metricas
