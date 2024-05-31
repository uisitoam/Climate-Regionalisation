import numpy as np 
from time import time
from bot import bot_texter
import reader as rd
from datafunctions import preprocess_data, reMap
from trainfunctions import entrenamiento, predicciones
from tensorflow import keras
from nets import temperatureModel, precipModel, gaussianLoss, bernouilliGammaLoss
from plots import boxplots, mapeo, animation, comparacion, big_boxplot





# reading Era5 data and importing all the processed data in reader.py
timeEra5, longitudeEra5, latitudeEra5, zEra5, qEra5, tEra5, uEra5, vEra5 = rd.reader("Datos/era5/data_era5ann_1982_2019_crop.nc", unit='hours', baseDate='19000101') # (13392, ), (17, ), (10,), (13392, 3, 10, 17)...
maskMap, maskBool = rd.maskMap, rd.maskBool # (158,), (68,), (1, 68, 158), (10744,)

# reading the data files (from 1982-01-05 to 2019-12-31)
tempTimeWRF, tMeanWRF = rd.reader("Datos/WRF/data_plcwrf_t2mean_1982_2019.nc", wrf=True) # (13392,), (13392, 68, 158)
precipTimeWRF, precipWRF = rd.reader("Datos/WRF/data_plcwrf_pr_1982_2019.nc", unit='minutes', baseDate='19810901', wrf=True) # (13392,), (13392, 68, 158)
precipTimeWRF2, precipWRF2 = rd.reader("Datos/WRF/data_plcwrf_pr_1982_2019_masked.nc", unit='minutes', baseDate='19810901', wrf=True) # (13392,), (13392, 68, 158)

tempTime_1980_2009, tempTime_2030_2059, tempTime_2070_2099 = rd.timeGFDL1_temp, rd.timeGFDL2_temp, rd.timeGFDL3_temp # (10571,), (10814,), (10886,)
precipTime_1980_2009, precipTime_2030_2059, precipTime_2070_2099 = rd.timeGFDL1_precip, rd.timeGFDL2_precip, rd.timeGFDL3_precip # (10231,), (9445,), (10895,)

tmeanWRF_GFDL1, tmeanWRF_IPSL1, tmeanWRF_MIROC1 = rd.tmeanWRF_GFDL1, rd.tmeanWRF_IPSL1, rd.tmeanWRF_MIROC1 # (10570, 68, 158), (10570, 68, 158), (10570, 68, 158)
tmeanWRF_GFDL2, tmeanWRF_IPSL2, tmeanWRF_MIROC2 = rd.tmeanWRF_GFDL2, rd.tmeanWRF_IPSL2, rd.tmeanWRF_MIROC2 # (10813, 68, 158), (10813, 68, 158), (10813, 68, 158)
tmeanWRF_GFDL3, tmeanWRF_IPSL3, tmeanWRF_MIROC3 = rd.tmeanWRF_GFDL3, rd.tmeanWRF_IPSL3, rd.tmeanWRF_MIROC3 # (10885, 68, 158), (10885, 68, 158), (10885, 68, 158)

precipWRF_GFDL1, precipWRF_IPSL1, precipWRF_MIROC1 = rd.precipWRF_GFDL1, rd.precipWRF_IPSL1, rd.precipWRF_MIROC1 # (10230, 68, 158), (10230, 68, 158), (10230, 68, 158)
precipWRF_GFDL2, precipWRF_IPSL2, precipWRF_MIROC2 = rd.precipWRF_GFDL2, rd.precipWRF_IPSL2, rd.precipWRF_MIROC2 # (9444, 68, 158), (9444, 68, 158), (9444, 68, 158)
precipWRF_GFDL3, precipWRF_IPSL3, precipWRF_MIROC3 = rd.precipWRF_GFDL3, rd.precipWRF_IPSL3, rd.precipWRF_MIROC3 # (10894, 68, 158), (10894, 68, 158), (10894, 68, 158)


# dividing the data for training and testing
div = list(timeEra5).index('20100101')


# prepare the sample data for the models
# era5 data (for training and testing)
era5_trainData = [zEra5[:div], qEra5[:div], tEra5[:div], uEra5[:div], vEra5[:div]]
era5_testData = [zEra5[div:], qEra5[div:], tEra5[div:], uEra5[div:], vEra5[div:]]
tempTimeWRF_train, tMeanWRF_train = tempTimeWRF[:div], tMeanWRF[:div] # (9751,), (9751, 68, 158)
tempTimeWRF_test, tMeanWRF_test = tempTimeWRF[div:], tMeanWRF[div:] # (3641,), (3641, 68, 158)
precipTimeWRF_train, precipWRF_train = precipTimeWRF[:div], precipWRF[:div] # (9751,), (9751, 68, 158)
precipTimeWRF_test, precipWRF_test = precipTimeWRF[div:], precipWRF[div:] # (3641,), (3641, 68, 158))

# GCMs temperature data (for testing)
GFDL1data_temp = [rd.zGFDL1_temp, rd.qGFDL1_temp, rd.tGFDL1_temp, rd.uGFDL1_temp, rd.vGFDL1_temp] # (5, 10570, 10, 17, 3)
IPSL1data_temp = [rd.zIPSL1_temp, rd.qIPSL1_temp, rd.tIPSL1_temp, rd.uIPSL1_temp, rd.vIPSL1_temp] # (5, 10570, 10, 17, 3)
MIROC1data_temp = [rd.zMIROC1_temp, rd.qMIROC1_temp, rd.tMIROC1_temp, rd.uMIROC1_temp, rd.vMIROC1_temp] # (5, 10570, 10, 17, 3)

GFDL2data_temp = [rd.zGFDL2_temp, rd.qGFDL2_temp, rd.tGFDL2_temp, rd.uGFDL2_temp, rd.vGFDL2_temp] # (5, 10813, 10, 17, 3)
IPSL2data_temp = [rd.zIPSL2_temp, rd.qIPSL2_temp, rd.tIPSL2_temp, rd.uIPSL2_temp, rd.vIPSL2_temp] # (5, 10813, 10, 17, 3)
MIROC2data_temp = [rd.zMIROC2_temp, rd.qMIROC2_temp, rd.tMIROC2_temp, rd.uMIROC2_temp, rd.vMIROC2_temp] # (5, 10813, 10, 17, 3)

GFDL3data_temp = [rd.zGFDL3_temp, rd.qGFDL3_temp, rd.tGFDL3_temp, rd.uGFDL3_temp, rd.vGFDL3_temp] # (5, 10885, 10, 17, 3)
IPSL3data_temp = [rd.zIPSL3_temp, rd.qIPSL3_temp, rd.tIPSL3_temp, rd.uIPSL3_temp, rd.vIPSL3_temp] # (5, 10885, 10, 17, 3)
MIROC3data_temp = [rd.zMIROC3_temp, rd.qMIROC3_temp, rd.tMIROC3_temp, rd.uMIROC3_temp, rd.vMIROC3_temp] # (5, 10885, 10, 17, 3)

# GCMs precipitation data (for testing)
GFDL1data_precip = [rd.zGFDL1_precip, rd.qGFDL1_precip, rd.tGFDL1_precip, rd.uGFDL1_precip, rd.vGFDL1_precip] # (5, 10230, 10, 17, 3)
IPSL1data_precip = [rd.zIPSL1_precip, rd.qIPSL1_precip, rd.tIPSL1_precip, rd.uIPSL1_precip, rd.vIPSL1_precip] # (5, 10230, 10, 17, 3)
MIROC1data_precip = [rd.zMIROC1_precip, rd.qMIROC1_precip, rd.tMIROC1_precip, rd.uMIROC1_precip, rd.vMIROC1_precip] # (5, 10230, 10, 17, 3)

GFDL2data_precip = [rd.zGFDL2_precip, rd.qGFDL2_precip, rd.tGFDL2_precip, rd.uGFDL2_precip, rd.vGFDL2_precip] # (5, 9444, 10, 17, 3)
IPSL2data_precip = [rd.zIPSL2_precip, rd.qIPSL2_precip, rd.tIPSL2_precip, rd.uIPSL2_precip, rd.vIPSL2_precip] # (5, 9444, 10, 17, 3)
MIROC2data_precip = [rd.zMIROC2_precip, rd.qMIROC2_precip, rd.tMIROC2_precip, rd.uMIROC2_precip, rd.vMIROC2_precip] # (5, 9444, 10, 17, 3)

GFDL3data_precip = [rd.zGFDL3_precip, rd.qGFDL3_precip, rd.tGFDL3_precip, rd.uGFDL3_precip, rd.vGFDL3_precip] # (5, 10894, 10, 17, 3)
IPSL3data_precip = [rd.zIPSL3_precip, rd.qIPSL3_precip, rd.tIPSL3_precip, rd.uIPSL3_precip, rd.vIPSL3_precip] # (5, 10894, 10, 17, 3)
MIROC3data_precip = [rd.zMIROC3_precip, rd.qMIROC3_precip, rd.tMIROC3_precip, rd.uMIROC3_precip, rd.vMIROC3_precip] # (5, 10894, 10, 17, 3)





# TRAINING THE MODELS

# TEMPERATURE
tempParams = [400, 128, 5]

tempEra5_trainSamples, temp_trainLabels, tempStore = preprocess_data(era5_trainData, tMeanWRF_train, maskBool, 'temp') # (9751, 10, 17, 15), (9751, 1059)
tempEra5_testSamples, temp_testLabels, tempStore = preprocess_data(era5_testData, tMeanWRF_test, maskBool, 'temp', store=tempStore, interpol=None) # (3641, 10, 17, 15), (3641, 1059)

# training the temperature model
# saves the models in the folder Resultados/Temperatura/modelos, which is created if it doesn't exist
#entrenamiento(temperatureModel, gaussianLoss, [tempEra5_trainSamples, temp_trainLabels], tempParams)


# PRECIPITATION
precipParams = [300, 128, 5]

precipEra5_trainSamples, precip_trainLabels, precipStore = preprocess_data(era5_trainData, precipWRF_train, maskBool, 'pr') 
precipEra5_testSamples, precip_testLabels, precipStore = preprocess_data(era5_testData, precipWRF_test, maskBool, 'pr', store=precipStore, interpol=None) 

# training the precipitation model
# saves the models in the folder Resultados/Precipitacion/modelos, which is created if it doesn't exist
#entrenamiento(precipModel, bernouilliGammaLoss, [precipEra5_trainSamples, precip_trainLabels], precipParams)





# RESULTS

def resultados(data, dates, store, modelo, modelos, mascara=rd.maskBool, trainProb=None,
               predicciones=predicciones, boxplot=boxplots, mapPaint=mapeo, mapeado=reMap, animacion = animation):
    """
    Function that calculates the predictions and the metrics for the temperature or precipitation data. 
    It gives the predictions and the metrics for the data and makes the boxplot, the map of them and the 
    animation.

    Parameters
    ----------
    data : list
        List with the training labels, the samples and the labels for the testing data.
    dates : list
        List with the dates of the testing data.
    store : dict
        Dictionary with the maximum and minimum values for the data.
    modelo : str
        Name of the model.
    modelos : list
        List with the models for the data.
    mascara : np.array  
        Mask for the islands.
    trainProb : np.array
        Probabilities of the training data.
    predicciones : function
        Function that calculates the predictions.
    boxplot : function
        Function that makes the boxplot.
    mapPaint : function
        Function that makes the map of the metrics.
    mapeado : function
        Function that maps the data.
    animacion : function
        Function that makes the animation.
    
    Returns
    -------
    predictions : np.array
        Array with the predictions of the data.
    metrics : np.array
        Array with the metrics of the data.

    """

    if trainProb is not None:
        var = 'pr'
        mets = 'prMets'
        title = 'Precipitation ' + modelo
        savePath = 'Resultados/Precipitacion/plots/' + modelo
        axLabels = ['Mean Bias', 'P98 Bias', 'R (Spearman)', 'RMSE (Wet Days)', 'Bias Wet AMS', 'Bias Dry AMS']

    else:
        var = 'temp'
        mets = 'tempMets'
        title = 'Temperature ' + modelo
        savePath = 'Resultados/Temperatura/plots/' + modelo
        axLabels = ['Mean Bias', 'P2 Bias', 'P98 Bias', 'R (Pearson)', 'std Ratio', 'RMSE', 'Bias WAMS', 'Bias CAMS']


    trainLabels, samples, labels = data
    predictions, metrics = predicciones(modelos, [trainLabels, samples, labels], store, dates, mascara, trainProb=trainProb)

    #bot_texter(f'Medianas del modelo {modelo}: {np.round(np.median(metrics, axis=1), 4)}')
    #bot_texter(f'Medias del modelo {modelo}: {np.round(np.mean(metrics, axis=1), 4)}')
    #bot_texter(f'Desv. estandar del modelo {modelo}: {np.round(np.std(metrics, axis=1), 4)}')

    predictions_map, metrics_map = mapeado(predictions, mascara), mapeado(metrics, mascara)

    q1 = np.nanquantile(metrics, 0.25, axis=1)
    q3 = np.nanquantile(metrics, 0.75, axis=1)
    iqr = q3 - q1
    
    boxplot(metrics, axLabels, [var, 9, 5, title, savePath, f'metricas{modelo}.pdf'])

    for i in range(len(axLabels)):
        cotas = [q1[i] - 1.5*iqr[i], q3[i] + 1.5*iqr[i]]

        nombre = axLabels[i].replace(' ', '_')
        mapPaint(metrics_map, dates, i, [mets, axLabels[i] + ' ' + modelo, savePath, f'{nombre}_map{modelo}.pdf'], cotas=cotas)
    
    #animacion(predictions_map, dates, [var, savePath + '/anim', 'mapita'])

    return predictions, metrics





# TEMPERATURE

# samples and labels for predicting the temperature with GCMs models and comparing them with the WRF data 
# 1980-2009
tempGFDL1_samples, tempGFDL1_labels, tempStore = preprocess_data(GFDL1data_temp, tmeanWRF_GFDL1, maskBool, 'temp', store=tempStore, interpol=None) # (10571, 10, 17, 15) (10571, 1059)
tempIPSL1_samples, tempIPSL1_labels, tempStore = preprocess_data(IPSL1data_temp, tmeanWRF_IPSL1, maskBool, 'temp', store=tempStore, interpol=None) # (10571, 10, 17, 15) (10571, 1059)
tempMIROC1_samples, tempMIROC1_labels, tempStore = preprocess_data(MIROC1data_temp, tmeanWRF_MIROC1, maskBool, 'temp', store=tempStore, interpol=None) # (10571, 10, 17, 15) (10571, 1059)

# 2030-2059
tempGFDL2_samples, tempGFDL2_labels, tempStore = preprocess_data(GFDL2data_temp, tmeanWRF_GFDL2, maskBool, 'temp', store=tempStore, interpol=None) # (10814, 10, 17, 15) (10814, 1059)
tempIPSL2_samples, tempIPSL2_labels, tempStore = preprocess_data(IPSL2data_temp, tmeanWRF_IPSL2, maskBool, 'temp', store=tempStore, interpol=None) # (10814, 10, 17, 15) (10814, 1059)
tempMIROC2_samples, tempMIROC2_labels, tempStore = preprocess_data(MIROC2data_temp, tmeanWRF_MIROC2, maskBool, 'temp', store=tempStore, interpol=None) # (10814, 10, 17, 15) (10814, 1059)

# 2070-2099
tempGFDL3_samples, tempGFDL3_labels, tempStore = preprocess_data(GFDL3data_temp, tmeanWRF_GFDL3, maskBool, 'temp', store=tempStore, interpol=None) # (10886, 10, 17, 15) (10814, 1059)
tempIPSL3_samples, tempIPSL3_labels, tempStore = preprocess_data(IPSL3data_temp, tmeanWRF_IPSL3, maskBool, 'temp', store=tempStore, interpol=None) # (10886, 10, 17, 15) (10814, 1059)
tempMIROC3_samples, tempMIROC3_labels, tempStore = preprocess_data(MIROC3data_temp, tmeanWRF_MIROC3, maskBool, 'temp', store=tempStore, interpol=None) # (10886, 10, 17, 15) (10814, 1059)


# loading the trained temperature models
tempModelos = [0] * tempParams[2]

for i in range(tempParams[2]):
    print(f'Loading temperature model {i+1}')
    tempModelos[i] = keras.models.load_model(f'./Resultados/Temperatura/modelos/temp({i}).keras', custom_objects={'gaussianLoss': gaussianLoss})


# making the predictions and calculating the results
# era5 data
tempEra5_predictions, tempEra5_metrics = resultados([temp_trainLabels, tempEra5_testSamples, temp_testLabels], timeEra5[div:], tempStore, 'Era5', tempModelos, trainProb=None)

# GCMs data (1980-2009)
tempGFDL1_predictions, tempGFDL1_metrics = resultados([temp_trainLabels, tempGFDL1_samples, tempGFDL1_labels], tempTime_1980_2009, tempStore, 'GFDL1', tempModelos, trainProb=None)
tempIPSL1_predictions, tempIPSL1_metrics = resultados([temp_trainLabels, tempIPSL1_samples, tempIPSL1_labels], tempTime_1980_2009, tempStore, 'IPSL1', tempModelos, trainProb=None)
tempMIROC1_predictions, tempMIROC1_metrics = resultados([temp_trainLabels, tempMIROC1_samples, tempMIROC1_labels], tempTime_1980_2009, tempStore, 'MIROC1', tempModelos, trainProb=None)

# GCMs data (2030-2059)
tempGFDL2_predictions, tempGFDL2_metrics = resultados([temp_trainLabels, tempGFDL2_samples, tempGFDL2_labels], tempTime_2030_2059, tempStore, 'GFDL2', tempModelos, trainProb=None)
tempIPSL2_predictions, tempIPSL2_metrics = resultados([temp_trainLabels, tempIPSL2_samples, tempIPSL2_labels], tempTime_2030_2059, tempStore, 'IPSL2', tempModelos, trainProb=None)
tempMIROC2_predictions, tempMIROC2_metrics = resultados([temp_trainLabels, tempMIROC2_samples, tempMIROC2_labels], tempTime_2030_2059, tempStore, 'MIROC2', tempModelos, trainProb=None)

# GCMs data (2070-2099)
tempGFDL3_predictions, tempGFDL3_metrics = resultados([temp_trainLabels, tempGFDL3_samples, tempGFDL3_labels], tempTime_2070_2099, tempStore, 'GFDL3', tempModelos, trainProb=None)
tempIPSL3_predictions, tempIPSL3_metrics = resultados([temp_trainLabels, tempIPSL3_samples, tempIPSL3_labels], tempTime_2070_2099, tempStore, 'IPSL3', tempModelos, trainProb=None)
tempMIROC3_predictions, tempMIROC3_metrics = resultados([temp_trainLabels, tempMIROC3_samples, tempMIROC3_labels], tempTime_2070_2099, tempStore, 'MIROC3', tempModelos, trainProb=None)


# comparing the results
comparacion([tempGFDL1_metrics, tempIPSL1_metrics, tempMIROC1_metrics, tempEra5_metrics,
                tempGFDL2_metrics, tempIPSL2_metrics, tempMIROC2_metrics, tempGFDL3_metrics, 
                tempIPSL3_metrics, tempMIROC3_metrics], 
                ['1980-2099', 'Resultados/Temperatura/plots/all_v1', 'all1980_2099.pdf'])

# making the big boxplot
big_boxplot([tempGFDL1_labels, tempGFDL1_predictions, tempIPSL1_labels, tempIPSL1_predictions, 
             tempMIROC1_labels, tempMIROC1_predictions, temp_testLabels, tempEra5_predictions,
             tempGFDL2_labels, tempGFDL2_predictions, tempIPSL2_labels, tempIPSL2_predictions,
             tempMIROC2_labels, tempMIROC2_predictions, tempGFDL3_labels, tempGFDL3_predictions,
             tempIPSL3_labels, tempIPSL3_predictions, tempMIROC3_labels, tempMIROC3_predictions], 'temp')





# PRECIPITATION

# samples and labels for predicting the precipitation with GCMs models and comparing them with the WRF data 
# 1980-2009
precipGFDL1_samples, precipGFDL1_labels, precipStore = preprocess_data(GFDL1data_precip, precipWRF_GFDL1, maskBool, 'pr', store=precipStore, interpol=None) # (10231, 10, 17, 15) (10231, 1059)
precipIPSL1_samples, precipIPSL1_labels, precipStore = preprocess_data(IPSL1data_precip, precipWRF_IPSL1, maskBool, 'pr', store=precipStore, interpol=None) # (10231, 10, 17, 15) (10231, 1059)
precipMIROC1_samples, precipMIROC1_labels, precipStore = preprocess_data(MIROC1data_precip, precipWRF_MIROC1, maskBool, 'pr', store=precipStore, interpol=None) # (10231, 10, 17, 15) (10231, 1059)

# 2030-2059
precipGFDL2_samples, precipGFDL2_labels, precipStore = preprocess_data(GFDL2data_precip, precipWRF_GFDL2, maskBool, 'pr', store=precipStore, interpol=None) # (9445, 10, 17, 15) (9445, 1059)
precipIPSL2_samples, precipIPSL2_labels, precipStore = preprocess_data(IPSL2data_precip, precipWRF_IPSL2, maskBool, 'pr', store=precipStore, interpol=None) # (9445, 10, 17, 15) (9445, 1059)
precipMIROC2_samples, precipMIROC2_labels, precipStore = preprocess_data(MIROC2data_precip, precipWRF_MIROC2, maskBool, 'pr', store=precipStore, interpol=None) # (9445, 10, 17, 15) (9445, 1059)

# 2070-2099
precipGFDL3_samples, precipGFDL3_labels, precipStore = preprocess_data(GFDL3data_precip, precipWRF_GFDL3, maskBool, 'pr', store=precipStore, interpol=None) # (10895, 10, 17, 15) (10895, 1059)
precipIPSL3_samples, precipIPSL3_labels, precipStore = preprocess_data(IPSL3data_precip, precipWRF_IPSL3, maskBool, 'pr', store=precipStore, interpol=None) # (10895, 10, 17, 15) (10895, 1059)
precipMIROC3_samples, precipMIROC3_labels, precipStore = preprocess_data(MIROC3data_precip, precipWRF_MIROC3, maskBool, 'pr', store=precipStore, interpol=None) # (10895, 10, 17, 15) (10895, 1059)


# loading the trained precipitation models
precipModelos = [0] * precipParams[2]
trainProbs = [0] * precipParams[2]

for i in range(precipParams[2]):
    print(f'Loading precipitation model {i+1}')
    precipModelos[i] = keras.models.load_model(f'./Resultados/Precipitacion/modelos/precip({i}).keras', custom_objects={'bernouilliGammaLoss': bernouilliGammaLoss})
    trainProbs[i] = np.load(f'./Resultados/Precipitacion/datos/precipTrainProb({i}).npy')


#making the predicions and calculating the results
# era5 data
precipEra5_predictions, precipEra5_metrics = resultados([precip_trainLabels, precipEra5_testSamples, precip_testLabels], timeEra5[div:], precipStore, 'Era5', precipModelos, trainProb=trainProbs)

# GCMs data (1980-2009)
precipGFDL1_predictions, precipGFDL1_metrics = resultados([precip_trainLabels, precipGFDL1_samples, precipGFDL1_labels], precipTime_1980_2009, precipStore, 'GFDL1', precipModelos, trainProb=trainProbs)
precipIPSL1_predictions, precipIPSL1_metrics = resultados([precip_trainLabels, precipIPSL1_samples, precipIPSL1_labels], precipTime_1980_2009, precipStore, 'IPSL1', precipModelos, trainProb=trainProbs)
precipMIROC1_predictions, precipMIROC1_metrics = resultados([precip_trainLabels, precipMIROC1_samples, precipMIROC1_labels], precipTime_1980_2009, precipStore, 'MIROC1', precipModelos, trainProb=trainProbs)

# GCMs data (2030-2059)
precipGFDL2_predictions, precipGFDL2_metrics = resultados([precip_trainLabels, precipGFDL2_samples, precipGFDL2_labels], precipTime_2030_2059, precipStore, 'GFDL2', precipModelos, trainProb=trainProbs)
precipIPSL2_predictions, precipIPSL2_metrics = resultados([precip_trainLabels, precipIPSL2_samples, precipIPSL2_labels], precipTime_2030_2059, precipStore, 'IPSL2', precipModelos, trainProb=trainProbs)
precipMIROC2_predictions, precipMIROC2_metrics = resultados([precip_trainLabels, precipMIROC2_samples, precipMIROC2_labels], precipTime_2030_2059, precipStore, 'MIROC2', precipModelos, trainProb=trainProbs)

# GCMs data (2070-2099)
precipGFDL3_predictions, precipGFDL3_metrics = resultados([precip_trainLabels, precipGFDL3_samples, precipGFDL3_labels], precipTime_2070_2099, precipStore, 'GFDL3', precipModelos, trainProb=trainProbs)
precipIPSL3_predictions, precipIPSL3_metrics = resultados([precip_trainLabels, precipIPSL3_samples, precipIPSL3_labels], precipTime_2070_2099, precipStore, 'IPSL3', precipModelos, trainProb=trainProbs)
precipMIROC3_predictions, precipMIROC3_metrics = resultados([precip_trainLabels, precipMIROC3_samples, precipMIROC3_labels], precipTime_2070_2099, precipStore, 'MIROC3', precipModelos, trainProb=trainProbs)


# comparing the results
comparacion([precipGFDL1_metrics, precipIPSL1_metrics, precipMIROC1_metrics, precipEra5_metrics, 
                precipGFDL2_metrics, precipIPSL2_metrics, precipMIROC2_metrics, precipGFDL3_metrics,
                precipIPSL3_metrics, precipMIROC3_metrics], 
                ['1980-2099', 'Resultados/Precipitacion/plots/all_v1', 'all1980_2099.pdf'])

# making the big boxplot
big_boxplot([precipGFDL1_labels, precipGFDL1_predictions, precipIPSL1_labels, precipIPSL1_predictions, 
             precipMIROC1_labels, precipMIROC1_predictions, precip_testLabels, precipEra5_predictions, 
             precipGFDL2_labels, precipGFDL2_predictions, precipIPSL2_labels, precipIPSL2_predictions,
             precipMIROC2_labels, precipMIROC2_predictions, precipGFDL3_labels, precipGFDL3_predictions,
             precipIPSL3_labels, precipIPSL3_predictions, precipMIROC3_labels, precipMIROC3_predictions], 'pr')
