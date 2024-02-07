import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from tensorflow.keras.optimizers.legacy import Adam
from sklearn.metrics import roc_auc_score
from nets import temperatureModel, precModel

colores = ['#FF0000', '#FF9B00', '#F5DE00', '#51FF53', '#00A902', '#00FF8F', 
           '#00F5E6', '#00B2FF', '#0032FF', '#B900FF', '#8700FF', '#FF86F6', 
           '#EA00D9', '#B00070', '#A4A4A4']


# split data into train and test subsets (samples and labels)
def interpolNsplit(index1, index2, minimo, maximo, store, name, samples=None, labels=None, masked=None):
    if samples is not None:
        samples2 = samples[:index2, :, :, :] # take all data
        if masked is not None: #temperature
            samples3 = np.moveaxis(samples2[masked, :, :, :], 1, -1) #move level axis to the last one 
        else: #precipitation
            samples3 = np.moveaxis(samples2, 1, -1) #move level axis to the last one 
            
        store[name] = [np.min(samples3), np.max(samples3)] # save min and max values (to then "desinterpol" data)
        nSamples = np.interp(samples3, (np.min(samples3), np.max(samples3)), (minimo, maximo)) # interpol data (normalize it or move it to [-1, 1] interval)
        train = nSamples[:index1, :, :, :] #train data
        test = nSamples[index1:, :, :, :] #test data 
        return train, test
    
    if labels is not None:
        nLabels = np.interp(labels, (store[name][0], store[name][1]), (minimo, maximo))
        train = nLabels[:index1]
        test = nLabels[index1:]
        return train, test
        
# "desinterpol" data to its original interval 
def denormalize(pred, storeMin, storeMax):
    denormalize = np.interp(pred, (0, 1), (storeMin, storeMax))
    return denormalize

# train the model and obtain results (saving loss and validation data on history variable)
def trainNpred(modelo, trainSamples, trainLabels, testSamples, denorm, store, name, ep, bs, lossfunc):
    model = modelo(trainSamples, trainLabels)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=lossfunc)
    model.summary()

    #train and store the data
    history = model.fit(trainSamples, trainLabels, validation_split=0.1, epochs=ep, batch_size=bs, verbose=1)

    predictions = model.predict(testSamples)
    denormalized_predictions = denorm(predictions, store[name][0], store[name][1])
    return history, denormalized_predictions


#obtain coefficients to see how good the model is performing 
#(mostly for temperature (besides Bias (warm annual max spell, WAMS) and 
#Bias (cold annual max spell, CAMS), yet to be done), specific precipitations coefficients are 
#not implemented yet: bias mean, bias p98, rmse (not ratio), 
#Spearman correlation, ROC skill score ROCSS, 
#Bias (wet annual max spell, WetAMS), Bias (dry annual max spell, DryAMS))

def coeficientes(predict, labels, outs, var):
    biasMean = np.zeros(outs)
    biasP98 = np.zeros(outs)    
    rmse = np.zeros(outs)

    def rootmse(preds, targets):
            return np.sqrt(((preds - targets) ** 2).mean())
    

    #funcion para contar días frios o calidos o para contar dias lluviosos o de sequia 

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
            
            bias_WAMS[i] = int(counting(labels[:, i], 'greater', 20.0) - counting(predict[:, i], 'greater', 20.0))
            bias_CAMS[i] = int(counting(labels[:, i], 'lower', 20.0) - counting(predict[:, i], 'lower', 20.0))
            

        
        return biasMean, biasP2, biasP98, pcorr, std_predictions, rmse, bias_WAMS, bias_CAMS
        
    if var == 'precipitation':
        spcorr = np.zeros(outs)
        #ROCSS = np.zeros(outs) 
        bias_WetAMS = np.zeros(outs)
        bias_DryAMS = np.zeros(outs)
        
        
        
        #Spearman
        #The Spearman rank-order correlation coefficient is a nonparametric measure 
        #of the monotonicity of the relationship between two datasets. Like other 
        #correlation coefficients, this one varies between -1 and +1 with 0 implying 
        #no correlation. Correlations of -1 or +1 imply an exact monotonic relationship. 
        #Positive correlations imply that as x increases, so does y. Negative correlations 
        #imply that as x increases, y decreases.

        #The Relative Operating Characteristics skill score (ROCSS), 
        #compares the predicted probability against the frequency with which the 
        #forecasts verify. ROCSS (ROC Skill Score) is a metric used to compare the 
        #performance of two different models based on their ROC curves. It measures 
        #the improvement of one model's ROC curve over another model's ROC curve. 
        #The ROCSS ranges from -1 to 1, with a positive value indicating improvement 
        #and a negative value indicating deterioration in performance.
        
        
        for i in range(outs):
            biasMean[i] = np.mean(labels[:, i]) - np.mean(predict[:, i])
            biasP98[i] = np.percentile(labels[:, i], 98) - np.percentile(predict[:, i], 98)
            
            spcorr[i] = spearmanr(predict[:, i].flatten(), labels[:, i].flatten())[0] #1d arrays
            #ROCSS[i] = roc_auc_score(true, score)
            rmse[i] = rootmse(predict[:, i], labels[:, i])
            
            bias_WetAMS[i] = int(counting(labels[:, i], 'greater', 0.0) - counting(predict[:, i], 'greater', 0.0))
            bias_DryAMS[i] = int(counting(labels[:, i], 'lower', 0.0) - counting(predict[:, i], 'lower', 0.0))
            

        
        return biasMean, biasP98, spcorr, rmse, bias_WetAMS, bias_DryAMS
    


#make label data (for train and test) (with a mask for temperature to be higher than 220K (to avoid bad readings), 
#and for precipitations to be higher or equal to 1 mm (under this is not considered as 
#rainy day)). Also obtain train a test samples from the era5 data of z, q, t, u and v. 


def extractData(era5, labs, trainTime, testTime, cte, cota=None):
    
    stTFN = labs[0][:testTime] + cte #273.15 for temperature, -1 for precipitation
    stTFS = labs[1][:testTime] + cte 
    stIZ = labs[2][:testTime] + cte 
    stSC = labs[3][:testTime] + cte 
    
    if cota is not None:
        mask = (stTFN >= cota) & (stIZ >= cota) & (stSC >= cota) # 220 for temperature
        
        labels = np.concatenate((stTFN[mask].reshape(-1, 1), stIZ[mask].reshape(-1, 1), 
                                 stSC[mask].reshape(-1, 1)), axis=1)
    else: 
        mask = None
        
        labels = np.concatenate((stTFN.reshape(-1, 1), stIZ.reshape(-1, 1), 
                                 stSC.reshape(-1, 1)), axis=1)
        
        labels[labels < 0] = 0
        
    minNmax = {}
    
    minNmax['labels(t)'] = [np.min(labels), np.max(labels)] # to interpol and "desinterpol" label data
    
    # data from era5 used to train and predict
    zTrain_samples, zTest_samples = interpolNsplit(trainTime, testTime, 
                                                   0, 1, minNmax, 'z', era5[0], None, mask)
    
    qTrain_samples, qTest_samples = interpolNsplit(trainTime, testTime, 
                                                   0, 1, minNmax, 'q', era5[1], None, mask)
    
    tTrain_samples, tTest_samples = interpolNsplit(trainTime, testTime, 
                                                   0, 1, minNmax, 't', era5[2], None, mask)
    
    uTrain_samples, uTest_samples = interpolNsplit(trainTime, testTime, 
                                                   0, 1, minNmax, 'u', era5[3], None, mask)
    
    vTrain_samples, vTest_samples = interpolNsplit(trainTime, testTime, 
                                                   0, 1, minNmax, 'v', era5[4], None, mask)

    #train and test data
    entreno = np.concatenate((zTrain_samples, qTrain_samples, tTrain_samples, uTrain_samples, vTrain_samples), axis=3)
    
    testeo = np.concatenate((zTest_samples, qTest_samples, tTest_samples, uTest_samples, vTest_samples), axis=3)
    
    # train and test labels 
    etiquetas, etiquetasTesteo = interpolNsplit(trainTime, testTime, 0, 1, 
                               minNmax, 'labels(t)', None, labels, mask)
        
    return entreno, etiquetas, testeo, etiquetasTesteo, minNmax


#execute the script and make the training curves and dataframes with the coefficients
def execution(model, data, loss, store, stations, columns, rows, epochs, repeats):
    
    if type(epochs) == int or type(epochs) == float:
        epochs = [epochs]
        

    for ep in epochs:
        train_loss = 0
        train_val = 0
        res = 0
        
        predLabelsDen = denormalize(data[3], store['labels(t)'][0], store['labels(t)'][1])
        
        biasMean = np.zeros(repeats)
        biasP98 = np.zeros(repeats)
        rmse = np.zeros(repeats)
        
        if model == temperatureModel:
            
            #temp_coefs = [np.zeros(repeats)] * 8
            
            biasP2 = np.zeros(repeats)
            pcorr = np.zeros(repeats)
            stdratio = np.zeros(repeats)
            bias_WAMS = np.zeros(repeats)
            bias_CAMS = np.zeros(repeats)
            
            temp_coefs = [biasMean, biasP2, biasP98, pcorr, stdratio, rmse, bias_WAMS, bias_CAMS]
            prec_coefs = 0
            
            for i in range(repeats):
                resultado = trainNpred(model, data[0], data[1], data[2], denormalize, 
                                       store, 'labels(t)', ep, 100, loss)

                train_loss += np.array(resultado[0].history['loss'])
                train_val += np.array(resultado[0].history['val_loss'])
                res += resultado[1]
                
                coefs = coeficientes(resultado[1][:, :stations], predLabelsDen, stations, 'temperature') #6 productos
                
                for j in range(len(temp_coefs)):
                    temp_coefs[j][i] = np.mean(coefs[j])
            
            
        elif model == precModel:
            
            #prec_coefs = [np.zeros(repeats)] * 6
            spcorr = np.zeros(repeats)
            bias_WetAMS = np.zeros(repeats)
            bias_DryAMS = np.zeros(repeats)
            
            prec_coefs = [biasMean, biasP98, spcorr, rmse, bias_WetAMS, bias_DryAMS]
            temp_coefs = 0
            
            for i in range(repeats):
                resultado = trainNpred(model, data[0], data[1], data[2], denormalize, 
                                       store, 'labels(t)', ep, 100, loss)
    
                train_loss += np.array(resultado[0].history['loss'])
                train_val += np.array(resultado[0].history['val_loss'])
                res += resultado[1]
                        
                coefs = coeficientes(resultado[1][:, :stations], predLabelsDen, stations, 'precipitation') #6 productos
                
                for j in range(len(prec_coefs)):
                    prec_coefs[j][i] = np.mean(coefs[j])
                    
                


        predictions = res/repeats
        training_loss = train_loss/repeats
        training_val = train_val/repeats
        
        
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
        
        df = pd.DataFrame(coefs, index=columns, columns=rows)

        df_d = df.style \
        .format(precision=4, decimal=",") \
        .format_index(str.upper, axis=1)
        
        #bot_texter('mira', file_data=None, file_name=None, data=df_d, name=f'{ep}epochs')
        """
        return predictions, temp_coefs, prec_coefs