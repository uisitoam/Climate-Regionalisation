import numpy as np 
import time as timeMod
from bot import timecalc, bot_texter, printeo
from reader import reader
from datafunctions import extractData, unmask, extractMask
from trainfunctions import entrenamiento, predicciones
from nets import temperatureModel, precipModel, gaussianLoss, bernouilliGammaLoss
from plots import boxplots, get_colormap, mapeo, animation
from tensorflow import keras


# reading the data files (from 1982-01-05 to 2019-12-31)
tDate, tMean, tLong, tLat = reader("Datos/WRF/data_plcwrf_t2mean_1982_2019.nc")
pDate, precip, pLong, pLat = reader("Datos/WRF/data_plcwrf_pr_1982_2019.nc")
pDate2, precip2, pLong2, pLat2 = reader("Datos/WRF/data_plcwrf_pr_1982_2019_masked.nc")

# reading the data from era5 file (from 1982-01-05 to 2019-12-31)
time, longitude, latitude, level, z, q, t, u, v = reader("Datos/era5/data_era5ann_1982_2019_crop.nc", 
                                                         era5=True)

# inicio en 19800101
timeGFDL1, zGFDL1, qGFDL1, tGFDL1, uGFDL1, vGFDL1 = reader("Datos/GCMs/all_GFDL_crop_1980_2009.nc")
timeGFDL2, zGFDL2, qGFDL2, tGFDL2, uGFDL2, vGFDL2 = reader("Datos/GCMs/all_GFDL_crop_2030_2059.nc")
timeGFDL3, zGFDL3, qGFDL3, tGFDL3, uGFDL3, vGFDL3 = reader("Datos/GCMs/all_GFDL_crop_2070_2099.nc")

timeIPSL1, zIPSL1, qIPSL1, tIPSL1, uIPSL1, vIPSL1 = reader("Datos/GCMs/all_IPSL_crop_1980_2009.nc")
timeIPSL2, zIPSL2, qIPSL2, tIPSL2, uIPSL2, vIPSL2 = reader("Datos/GCMs/all_IPSL_crop_2030_2059.nc")
timeIPSL3, zIPSL3, qIPSL3, tIPSL3, uIPSL3, vIPSL3 = reader("Datos/GCMs/all_IPSL_crop_2070_2099.nc")

timeMIROC1, zMIROC1, qMIROC1, tMIROC1, uMIROC1, vMIROC1 = reader("Datos/GCMs/all_MIROC_crop_1980_2009.nc") #estos empiezan en febrero 
timeMIROC2, zMIROC2, qMIROC2, tMIROC2, uMIROC2, vMIROC2 = reader("Datos/GCMs/all_MIROC_crop_2030_2059.nc")
timeMIROC3, zMIROC3, qMIROC3, tMIROC3, uMIROC3, vMIROC3 = reader("Datos/GCMs/all_MIROC_crop_2070_2099.nc")

times = [time, time.index('20100101'), time.index('20191231')]

era5data = [z, q, t, u, v]





# TEMPERATURE
tempParams = [400, 128, 5]

# temperature data (in Kelvin)
tempData, tempStore, tempTime = extractData(era5data, tMean, times, "Temperature")

# temperature results
start2 = timeMod.time()
entrenamiento(temperatureModel, gaussianLoss, tempData[:2], tempParams)

tempModelos = [0] * tempParams[2]

for i in range(tempParams[2]):
    print(f'Loading temperature model {i+1}')
    tempModelos[i] = keras.models.load_model(f'./Resultados/Temperatura/modelos/temp({i}).keras', custom_objects={'gaussianLoss': gaussianLoss})

tempPred, tempMetricas, tempMeans = predicciones(tempModelos, tempData, tempStore, tempTime[1])

fin2 = timeMod.time()
timecalc(start2, fin2, f'Tiempo para el modelo de temperatura ({tempParams[0]} epochs, batch size de {tempParams[1]} y {tempParams[2]} repeticiones)', True)

bot_texter(f'Medianas: {np.median(tempMetricas, axis=1)}')
bot_texter(f'Medias: {np.mean(tempMeans, axis=0)}')
bot_texter(f'Std: {np.std(tempMeans, axis=0)}')


# temperature boxplot
tCols = ['Mean Bias', 'P2 Bias', 'P98 Bias', 'R (Pearson)', 'std Ratio', 'RMSE', 'Bias WAMS', 'Bias CAMS']

boxplots(tempMetricas, tCols, 
         ['Temperature', f'./Resultados/Temperatura/plots/{tempParams[0]}_epochsTempMetrics.pdf'])

# temperature animation map (over 2017)
animation(tempPred - 273.15, [np.min(tLong), np.max(tLong), np.min(tLat), np.max(tLat)], 
          't', tempTime[1][2546:], 365*3)

# temperature metrics maps
for i in range(len(tCols)):
    mapeo(tempMetricas, [np.shape(tempMetricas)[0], 68, 158], 
      [np.min(tLong), np.max(tLong), np.min(tLat), np.max(tLat)], 
      [get_colormap('tm'), f'./Resultados/Temperatura/plots/tempMetricsMap{i}.pdf'], 
      [0, i])





# PRECIPITATION
precipParams = [300, 32, 5]

# precipitation data (without and with a sea mask respectively)
precipData, precipStore, precipTime = extractData(era5data, precip - 1, times, "Precipitation")
precipData2 = [precipData[0], extractMask(precip2[:times[1]]), precipData[2], extractMask(precip2[times[1]:])]


# precipitation results (using only values over the islands)
start4 = timeMod.time()
entrenamiento(precipModel, bernouilliGammaLoss, precipData2[:2], precipParams)

precipModelos = [0] * precipParams[2]
trainProbs = [0] * precipParams[2]

for i in range(precipParams[2]):
    print(f'Loading precipitation model {i+1}')
    precipModelos[i] = keras.models.load_model(f'./Resultados/Precipitacion/modelos/precip({i}).keras', custom_objects={'bernouilliGammaLoss': bernouilliGammaLoss})
    trainProbs[i] = np.load(f'./Resultados/Precipitacion/datos/precipTrainProb({i}).npy')

precipPred, precipMetricas, precipMeans = predicciones(precipModelos, precipData2, precipStore, 
                                                       precipTime[1], trainProbs)

fin4 = timeMod.time()
timecalc(start4, fin4, f'Tiempo para el modelo de precipitaciones ({precipParams[0]} epochs, batch size de {precipParams[1]} y {precipParams[2]} repeticiones)', True)

bot_texter(f'Medianas: {np.median(precipMetricas, axis=1)}')
bot_texter(f'Medias: {np.mean(precipMeans, axis=0)}')
bot_texter(f'Std: {np.std(precipMeans, axis=0)}')


# fill the complete grid with predicted values for islands
precipData3 = np.reshape(precipData[3], (np.shape(precipData[3])[0], 68*158))
reMask = np.reshape(precip2[times[1]:] >= 0, (np.shape(precip2[times[1]:] >= 0)[0], 68*158))

printeo('Construyendo resultados...')

precipPredValues = unmask(precipPred, np.where(reMask[:len(precipPred)] == False, 0, -999.))
precipMetricsValues = unmask(precipMetricas, np.where(reMask[:6, :] == False, 0, -999.))

# precipitation boxplots
pCols = ['Mean Bias', 'P98 Bias', 'R (Spearman)', 'RMSE (Wet days)', 'Bias WetAMS', 'Bias_DryAMS']

boxplots(precipMetricas, pCols, 
         ['Precipitation', f'./Resultados/Precipitacion/plots/(mask){precipParams[0]}_epochsPrecipMetrics.pdf'])

printeo('Sacando fotogramas...')

# precipitation animation map (over 2017)
animation(precipPredValues, [np.min(pLong), np.max(pLong), np.min(pLat), np.max(pLat)], 
          'p', precipTime[1][2546:], len(precipTime[1][2546:]))

# precipitation metrics maps
for i in range(len(pCols)):
    mapeo(precipMetricsValues, [np.shape(precipMetricsValues)[0], 68, 158], 
      [np.min(pLong), np.max(pLong), np.min(pLat), np.max(pLat)], 
      [get_colormap('pm'), f'./Resultados/Precipitacion/plots/(mask)precipMetricsMap{i}.pdf'], 
      [0, i], True)