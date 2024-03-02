import numpy as np
from nets import temperatureModel, precipModel
from datafunctions import coeficientes, computeRainfall
from plots import learningCurve
import matplotlib.pyplot as plt

#execute the script and make the training curves and dataframes with the coefficients
def execution(model, loss, data, store, times, params, stochastic = False):
    """
    This function execute the model and gives the predictions on given data.

    Parameters
    ----------
    model : function
        Select between temperature and precipitation model.
    loss : function
        Loss functions used to train the model: a gaussian for temperature and 
        a bernouilli gamma for precipitation.
    data : list
        Contains the arrays of data:first column is for train samples, second
        one for train labels, third one for test samples and last one for 
        test labels. 
    store : dict
        Contains min and max values of the original data.
    times : list
        Contains the dates of the data. 
    params : list
        Contains [epochs, batchzsize, repeats].
    stochastic : bool, optional
        If True, amount of rain is obtained stochastically from the bernouilli-
        gamma distribution. The default is False. 
        
    Raises
    ------
    ValueError
        - If `data` does not have the expected structure.
        - If `store` does not have the expected type.
        - If `times` is not a list of strings.
        - If `params` does not have the expected length.
    TypeError
        - If elements in `data` or `params` are not NumPy arrays.

    Returns
    -------
    predictions : np.ndarray
        Data predicted by the model. 
    metricas : list
        Metrics used to evaluate the models's performance.

    """
    
    if not isinstance(data, list) or len(data) != 4:
        raise ValueError("data must be a list of length 4 (train samples, labels, test samples, test labels)")
        
    if not all(isinstance(i, np.ndarray) for i in data):
        raise TypeError("Elements in data must be NumPy arrays")

    if not isinstance(store, dict):
        raise ValueError("store must be a dictionary")
       
    if not isinstance(times, list) or len(times) != 2 or not all(isinstance(t, str) for t in times[1]):
        raise ValueError("times must be a list of two string lists")
    
    if not isinstance(params, list) or len(params) != 3:
        raise ValueError("params must be a list of length 3 (epochs, batch_size, repeats)")


    train_loss = 0
    train_val = 0
    res = 0
    coefs = 0
    predProb = 0
    
    data1 = data[1].reshape((np.shape(data[1])[0], np.shape(data[1])[1]*np.shape(data[1])[2]))
    data3 = data[3].reshape((np.shape(data[3])[0], np.shape(data[3])[1]*np.shape(data[3])[2]))
     
    if model == temperatureModel:
        predLabelsDen = np.interp(data3, (0, 1), (store['labels(t)'][0], 
                                                    store['labels(t)'][1]))

        for i in range(params[2]):
            modelo, historial = model(data[0], data1, loss, params[:2])
            
            
            train_loss += np.array(historial.history['loss'])
            train_val += np.array(historial.history['val_loss'])
            
            preds = modelo.predict(data[2])
            predictions = preds[:, :int(np.shape(preds)[1]/2)]
            
            denormalized_predictions =  np.interp(predictions, (0, 1), 
                                                  (store['labels(t)'][0], 
                                                   store['labels(t)'][1]))
            
            res += denormalized_predictions
            coefs += coeficientes(denormalized_predictions, predLabelsDen, 
                                  times[1], 'temperature')

    elif model == precipModel:
        predLabelsDen = data3
        
        for i in range(params[2]):
            modelo, historial, trainProb = model(data[0], data1, loss, params[:2])
            
            train_loss += np.array(historial.history['loss'])
            train_val += np.array(historial.history['val_loss'])
            
            predictions = modelo.predict(data[2]) #prob, log_alpha, log_beta
            
            predProb += predictions[:, :int(np.shape(predictions)[1]/3)]
    
            precipAmount = computeRainfall([data[0], data1, trainProb], 
                                           predictions, stochastic)
            
            res += precipAmount
            coefs += coeficientes(precipAmount, predLabelsDen, 
                                  times[1], 'precipitation')
            

    metricas = coefs/params[2]
    predicts = res/params[2]
    #training_loss = train_loss/params[2]
    #training_val = train_val/params[2]
    training_loss = train_loss
    training_val = train_val
    
    if model == precipModel:
            precipProb = predProb/params[2] 
            
            figt, axt = plt.subplots(1, 3, figsize=(9,5))

            axt[0].hist(precipProb[:, 0])
            axt[1].hist(precipProb[:, 1])
            axt[2].hist(precipProb[:, 2])
            axt[0].set(xlabel='Rain probability',
                    ylabel='Frecuency', 
                    title='TFN')
            axt[1].set(xlabel='Rain probability',
                    ylabel='Frecuency', 
                    title='IZ')
            axt[2].set(xlabel='Rain probability',
                    ylabel='Frecuency', 
                    title='SC')
        
            #figt.savefig(f'./Resultados/Precipitacion/plots/{params[0]}_epochsHist.pdf')
    
    if model == temperatureModel:
        learningCurve([training_loss, training_val], params[0], 
                      f'./Resultados/Temperatura/plots/{params[0]}_epochsLearning_curve(temp).pdf')
    
    elif model == precipModel:
        learningCurve([training_loss, training_val], params[0], 
                      f'./Resultados/Precipitacion/plots/{params[0]}_epochsLearning_curve(precip).pdf')
    
    return predicts, metricas
