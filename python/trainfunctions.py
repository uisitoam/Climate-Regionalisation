import numpy as np
from nets import temperatureModel, precipModel
from datafunctions import coeficientes, computeRainfall
from plots import learningCurve


#execute the script and make the training curves and dataframes with the coefficients
def entrenamiento(model, loss, data, params):
    """
    This function train the model.

    Parameters
    ----------
    model : function
        Select between temperature and precipitation model.
    loss : function
        Loss functions used to train the model: a gaussian for temperature and 
        a bernouilli gamma for precipitation.
    data : list
        Contains the arrays of data: first column is for train samples, second
        one for train labels.
    params : list
        Contains [epochs, batchzsize, repeats].
        
    Raises
    ------
    ValueError
        - If `data` does not have the expected structure.
        - If `params` does not have the expected length.
    TypeError
        - If elements in `data` or `params` are not NumPy arrays.

    Returns
    -------
    None

    """
    
    if not isinstance(data, list) or len(data) != 2:
        raise ValueError("data must be a list of length 2 [train samples, train_labels]")
        
    if not all(isinstance(i, np.ndarray) for i in data):
        raise TypeError("Elements in data must be NumPy arrays")
    
    if not isinstance(params, list) or len(params) != 3:
        raise ValueError("params must be a list of length 3 (epochs, batch_size, repeats)")


    train_loss = 0
    train_val = 0
    samples, labels = data
    if model == temperatureModel:
        tLabels = data[1].reshape((np.shape(data[1])[0], np.shape(data[1])[1]*np.shape(data[1])[2]))

        for i in range(params[2]):
            modelo, historial = model(samples, tLabels, loss, params[:2])
            modelo.save(f'./Resultados/Temperatura/modelos/temp({i}).keras')
            
            train_loss += np.array(historial.history['loss'])
            train_val += np.array(historial.history['val_loss'])
            
    elif model == precipModel:
        
        for i in range(params[2]):
            modelo, historial, trainProb = model(samples, labels, loss, params[:2])
            modelo.save(f'./Resultados/Precipitacion/modelos/precip({i}).keras')

            np.save(f'./Resultados/Precipitacion/datos/precipTrainProb({i})', trainProb)
            
            train_loss += np.array(historial.history['loss'])
            train_val += np.array(historial.history['val_loss'])

    training_loss = train_loss/params[2]
    training_val = train_val/params[2]
   
    if model == temperatureModel:
        learningCurve([training_loss, training_val], 
                      f'./Resultados/Temperatura/plots/{params[0]}_epochsLearning_curve(temp).pdf')
    
    elif model == precipModel:
        learningCurve([training_loss, training_val], 
                      f'./Resultados/Precipitacion/plots/{params[0]}_epochsLearning_curve(precip).pdf')
    
    return
    




def predicciones(model, data, store, times, trainProb = None, stochastic = False):
    """
    This function gives the predictions of the model.

    Parameters
    ----------
    model : list 
        Trained models to predict with.
    data : list
        Contains the arrays of data:first column is for train samples, second
        one for train labels, third one for test samples and last one for 
        test labels. 
    store : dict
        Contains min and max values of the original data.
    times : list
        Contains the dates of the data. 
    trainProb : np.ndarray, optional
        Probability of occurrence for the train data of each model. It must be 
        given to make predictions on precipitation, not on temperature.
    stochastic : bool, optional
        If True, amount of rain is obtained stochastically from the bernouilli-
        gamma distribution. The default is False. 
        
    Raises
    ------
    ValueError
        - If `models`is not a list with the keras models.
        - If `data` does not have the expected structure.
        - If `store` does not have the expected type.
    TypeError
        - If elements in `data` are not NumPy arrays.

    Returns
    -------
    predicts : np.ndarray
        Data predicted by the model. 
    metricas : list
        Metrics used to evaluate the models's performance.
    allMeans : np.ndarray
        Mean results of the given models. 

    """

    if not isinstance(model, list):
        raise ValueError("model must be a list of results from the different models")
    
    if not isinstance(data, list) or len(data) != 4:
        raise ValueError("data must be a list of length 4 (train samples, labels, test samples, test labels)")
        
    if not all(isinstance(i, np.ndarray) for i in data):
        raise TypeError("Elements in data must be NumPy arrays")

    if not isinstance(store, dict):
        raise ValueError("store must be a dictionary")


    trainSamples, trainLabels, testSamples, test_labels = data
    res = 0
    coefs = 0
    predProb = 0
    medias = [0]*len(model)

    if trainProb is not None:
        for i in range(len(model)):
            predictions = model[i].predict(testSamples) #prob, log_alpha, log_beta
        
            predProb += predictions[:, :int(np.shape(predictions)[1]/3)]

            amountData, prob = computeRainfall([trainSamples, trainLabels, trainProb[i]], 
                                               predictions, stochastic)

            porcentaje = np.sum(trainLabels <= 0, axis=0)/np.shape(trainLabels)[0]
            cota = np.diag(np.nanquantile(trainProb[i], porcentaje, axis=0))           
            precipAmount = np.where(prob < cota, 0, amountData) #(condicion, True, False)
            
            res += precipAmount
            
            coefis = coeficientes(precipAmount, test_labels[:len(precipAmount)], 
                                  times, 'precipitation')
            
            coefs += coefis
            medias[i] = np.mean(coefis, axis=1)
     
    else:
        testLabels_norm = test_labels.reshape((np.shape(test_labels)[0], 
                                               np.shape(test_labels)[1]*np.shape(test_labels)[2]))
        
        testLabels = np.interp(testLabels_norm, (0, 1), (store['labels(t)'][0], 
                                                    store['labels(t)'][1]))

        for i in range(len(model)):
            preds = model[i].predict(testSamples)
            predictions_norm = preds[:, :int(np.shape(preds)[1]/2)]
        
            predictions =  np.interp(predictions_norm, (0, 1), 
                                                (store['labels(t)'][0], 
                                                store['labels(t)'][1]))
            
            res += predictions
            coefis = coeficientes(predictions, testLabels, 
                                  times, 'temperature')
            coefs += coefis

            medias[i] = np.mean(coefis, axis=1)
            
    metricas = coefs/len(model)
    predicts = res/len(model)
    allMeans = np.stack((medias))
    
    return predicts, metricas, allMeans
