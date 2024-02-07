import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, gamma
import tensorflow as tf
from tensorflow.keras.optimizers.legacy import Adam
from nets import temperatureModel, precipModel

colores = ['#FF0000', '#FF9B00', '#F5DE00', '#51FF53', '#00A902', '#00FF8F', 
           '#00F5E6', '#00B2FF', '#0032FF', '#B900FF', '#8700FF', '#FF86F6', 
           '#EA00D9', '#B00070', '#A4A4A4']


def trainNpred(modelo, lossfunc, data, store, params):
    """
    Train the model and obtain results (saving loss and validation data on 
    history variable).

    Parameters
    ----------
    modelo : function
        Model to use.
    lossfunc : function
        Loss functions to minimize during training.
    data : list
        Data feed to the model. 
    store : dict
        Contains min and max values of the original data.
    params : list
        Contains [epochs, batch size].

    Returns
    -------
    history : keras.src.callbacks.History
        History of training process.
    denormalized_predictions : np.ndarray
        Predictions on the original interval of values.

    """
    model = modelo(data[0], data[1])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=lossfunc)
    model.summary()

    #train and store the data
    history = model.fit(data[0], data[1], validation_split=0.1, 
                    epochs=params[0], batch_size=params[1], verbose=1)
    """
    if modelo == temperatureModel:
        history = model.fit(data[0], data[1], validation_split=0.1, 
                        epochs=params[0], batch_size=params[1], verbose=1, 
                        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3), 
                                   tf.keras.callbacks.ModelCheckpoint(filepath='./Temperatura/Modelos/CNN.h5', 
                                                   monitor='val_loss', save_best_only=True)])
    elif modelo == precipModel:
        history = model.fit(data[0], data[1], validation_split=0.1, 
                        epochs=params[0], batch_size=params[1], verbose=1, 
                        callbacks=[tf.keras.callbacks.EarlyStopping(patience=30), 
                                   tf.keras.callbacks.ModelCheckpoint(filepath='./Precipitacion/Modelos/CNN.h5', 
                                                   monitor='val_loss', save_best_only=True)])
    """    
    predictions = model.predict(data[2])
    
    if modelo == temperatureModel:
        denormalized_predictions =  np.interp(predictions, (0, 1), 
                                              (store['labels(t)'][0], 
                                               store['labels(t)'][1]))
        
        return history, denormalized_predictions
    
    else:
        return history, predictions





def coeficientes(predict, labels, outs, var):
    """
    Obtain metrics (coefficients) to see how good the model is performing. 
    - Temperature: Mean bias, percentile 2 and 98 bias, pearson correlation
    coefficient, standard deviation ratio, root mean square error and bias for 
    warm and cold annual max spells (WAMS and CAMS).
    - Precipitation: Mean bias, percentile 98 bias, spearson correlation 
    coefficient, root mean square error, bias for wet and dry annual max spells
    (WetAMS and DryAMS) and (yet to be implemented) ROC skill score (ROCSS).
    
    The Spearman rank-order correlation coefficient is a nonparametric measure 
    of the monotonicity of the relationship between two datasets. Like other 
    correlation coefficients, this one varies between -1 and +1 with 0 implying 
    no correlation. Correlations of -1 or +1 imply an exact monotonic 
    relationship. Positive correlations imply that as x increases, so does y. 
    Negative correlations imply that as x increases, y decreases.

    The Relative Operating Characteristics skill score (ROCSS), compares the 
    predicted probability against the frequency with which the forecasts 
    verify. ROCSS (ROC Skill Score) is a metric used to compare the performance 
    of two different models based on their ROC curves. It measures the 
    improvement of one model's ROC curve over another model's ROC curve. The 
    ROCSS ranges from -1 to 1, with a positive value indicating improvement 
    and a negative value indicating deterioration in performance.

    Parameters
    ----------
    predict : np.ndarray
        Predicted data.
    labels : np.ndarray
        True data.
    outs : int
        Stations with available labels.
    var : str
        Model we are dealing with, either temperature or precipitation.

    Returns
    -------
    np.ndarray
        Metrics to evaluate the model's performance. 
        - Temperature: Mean bias, P2 bias, P98 bias, Pearson correlation 
        coefficient, stardard deviation ratio, RMSE, warm annual max spell  bias, 
        cold annual max spell bias).
        - Precipitation: Mean bias, P98 bias, Spearman correlation coefficient, 
        RMSE, wet annual max spell bias, dry annual max spell bias.

    """
    
    biasMean = np.zeros(outs)
    biasP98 = np.zeros(outs)    
    rmse = np.zeros(outs)

    def rootmse(preds, targets):
        return np.sqrt(((preds - targets) ** 2).mean())
        
    #count max spells 
    def counting(arr, tipo, valor): #tipo es greater o lower 
        maximo = 0
        conteo = 0
        
        for num in arr:
            if tipo == 'lower':
                if num <= valor:
                    conteo += 1
                else: 
                    maximo = max(maximo, conteo)
                    conteo = 0
            elif tipo == 'greater': 
                if num > valor:
                    conteo += 1
                else: 
                    maximo = max(maximo, conteo)
                    conteo = 0
        
        return maximo
    
    
    
    if var == 'temperature':
        biasP2 = np.zeros(outs)
        pcorr = np.zeros(outs)
        std_predictions = np.zeros(outs)
        bias_WAMS = np.zeros(outs)
        bias_CAMS = np.zeros(outs)
    
        for i in range(outs):
            biasMean[i] = np.mean(labels[:, i]) - np.mean(predict[:, i])
            biasP2[i] = np.percentile(labels[:, i], 2) - np.percentile(predict[:, i], 2)
            biasP98[i] = np.percentile(labels[:, i], 98) - np.percentile(predict[:, i], 98)
            
            pcorr[i] = np.corrcoef(predict[:, i].flatten(), labels[:, i].flatten())[0,1]
            std_predictions[i] = np.std(predict[:, i])/np.std(labels[:, i])
            rmse[i] = rootmse(predict[:, i], labels[:, i])
            
            bias_WAMS[i] = int(counting(labels[:, i], 'greater', 20.0) 
                               - counting(predict[:, i], 'greater', 20.0))
            
            bias_CAMS[i] = int(counting(labels[:, i], 'lower', 20.0) 
                               - counting(predict[:, i], 'lower', 20.0))
            

        
        return biasMean, biasP2, biasP98, pcorr, std_predictions, rmse, bias_WAMS, bias_CAMS
        
    if var == 'precipitation':
        spcorr = np.zeros(outs)
        #ROCSS = np.zeros(outs) 
        bias_WetAMS = np.zeros(outs)
        bias_DryAMS = np.zeros(outs)
        
        
        for i in range(outs):
            biasMean[i] = np.mean(labels[:, i]) - np.mean(np.array(predict[:, i]))
            biasP98[i] = np.percentile(labels[:, i], 98) - np.percentile(np.array(predict[:, i]), 98)
            
            spcorr[i] = spearmanr(np.array(predict[:, i]), labels[:, i])[0] #1d arrays
            #ROCSS[i] = roc_auc_score(true, score)
            rmse[i] = rootmse(np.array(predict[:, i]), labels[:, i])
            
            bias_WetAMS[i] = int(counting(labels[:, i], 'greater', 0.0) 
                                 - counting(np.array(predict[:, i]), 'greater', 0.0))
            
            bias_DryAMS[i] = int(counting(labels[:, i], 'lower', 0.0) 
                                 - counting(np.array(predict[:, i]), 'lower', 0.0))
            

        
        return biasMean, biasP98, spcorr, rmse, bias_WetAMS, bias_DryAMS
    




#simulate = True es stochastic
def computeRainfall(log_alpha, log_beta, simulate = False, bias = None):
    
    if simulate:
        aux = np.zeros((np.shape(log_alpha)[0], np.shape(log_alpha)[1]))
        alpha = tf.exp(log_alpha)
        beta = tf.exp(log_beta)
        
        for i in range(np.shape(log_alpha)[1]):  
            aux[:, i] = gamma.rvs(alpha[:, i], beta[:, i], size=np.shape(log_alpha)[0])
        
        amountData = aux
    
    else:
        amountData = tf.exp(log_alpha)*tf.exp(log_beta) 
        
    if bias is not None:
        amountData += bias
        
    return amountData



#execute the script and make the training curves and dataframes with the coefficients
def execution(model, loss, data, store, params):
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
    params : list
        Contains [stations, epochs, batchzsize, repeats].

    Returns
    -------
    predictions : np.ndarray
        Data predicted by the model. 
    metricas : list
        Metrics used to evaluate the models's performance.

    """

    if type(params[1]) == int or type(params[1]) == float:
        params[1] = [params[1]]
        

    for ep in params[1]:
        train_loss = 0
        train_val = 0
        res = 0
                
        biasMean = np.zeros(params[3])
        biasP98 = np.zeros(params[3])
        rmse = np.zeros(params[3])
        
        if model == temperatureModel:
            
            #temp_coefs = [np.zeros(repeats)] * 8
            
            biasP2 = np.zeros(params[3])
            pcorr = np.zeros(params[3])
            stdratio = np.zeros(params[3])
            bias_WAMS = np.zeros(params[3])
            bias_CAMS = np.zeros(params[3])
            
            metricas = [biasMean, biasP2, biasP98, pcorr, stdratio, rmse, bias_WAMS, bias_CAMS]
            
            predLabelsDen = np.interp(data[3], (0, 1), (store['labels(t)'][0], store['labels(t)'][1]))

            #trainNpred(modelo, data, store, params, lossfunc, denorm=None):
            for i in range(params[3]):
                resultado = trainNpred(model, loss, data, store, [ep, params[2]])

                train_loss += np.array(resultado[0].history['loss'])
                train_val += np.array(resultado[0].history['val_loss'])
                res += resultado[1]
                
                coefs = coeficientes(resultado[1][:, :params[0]], predLabelsDen, 
                                     params[0], 'temperature') #6 productos
                
                for j in range(len(metricas)):
                    metricas[j][i] = np.mean(coefs[j])
            
            
        elif model == precipModel:
            
            #prec_coefs = [np.zeros(repeats)] * 6
            spcorr = np.zeros(params[3])
            bias_WetAMS = np.zeros(params[3])
            bias_DryAMS = np.zeros(params[3])
            
            metricas = [biasMean, biasP98, spcorr, rmse, bias_WetAMS, bias_DryAMS]
            
            predLabelsDen = data[3]

            
            for i in range(params[3]):
                resultado = trainNpred(model, loss, data, store, [ep, 100])
                
                train_loss += np.array(resultado[0].history['loss'])
                train_val += np.array(resultado[0].history['val_loss'])
                resultado1 = computeRainfall(resultado[1][:, 3:6], resultado[1][:, 6:], 
                                             simulate = False, bias = 1)
                res += resultado1
                #res += resultado[1]
                        
                coefs = coeficientes(resultado1[:, :params[0]], predLabelsDen, 
                                     params[0], 'precipitation') #6 productos
                
                for j in range(len(metricas)):
                    metricas[j][i] = np.mean(coefs[j])
                    
                


        predictions = res/params[3]
        training_loss = train_loss/params[3]
        training_val = train_val/params[3]
        
        
        fig, ax = plt.subplots(1, 1, figsize=(9,5))

        ejex = np.arange(1, ep + 1)

        ax.plot(ejex, training_loss, color=colores[5], label='Training Loss')
        ax.plot(ejex, training_val, color=colores[9], label='Validation Loss')
        ax.set(xlabel='Epochs',
                ylabel='Loss', 
                title='Training and Validation Loss')
        ax.legend()

        #fig.savefig(f'/Users/luisi/Library/Mobile Documents/com~apple~CloudDocs/Física/4. TFG/Códigos galácticos/GraficasCNN/{ep}_epochsLearning_curve.pdf')
        
        """
        #save the plot as a PNG image to a memory buffer
        buffer1 = io.BytesIO()
        fig.savefig(buffer1, format='png')
        buffer1.seek(0)

        #encode the image data as a base64 string
        image_base64_1 = base64.b64encode(buffer1.getvalue()).decode() 

        #bot_texter('mira', file_data=buffer1.getvalue(), file_name=f'{ep}_epochsLearning_curve.png')
        
        """

        return predictions, metricas
    