import numpy as np 
import time as timeMod
from bot import timecalc, printeo
from reader import reader
from datafunctions import extractData, unmask, extractMask
from trainfunctions import execution
from nets import temperatureModel, precipModel, gaussianLoss, bernouilliGammaLoss
from plots import boxplots, get_colormap, mapeo, animation
from tqdm import tqdm


# reading the data files (from 1982-01-05 to 2019-12-31)
tDate, tMean, tLong, tLat = reader("Datos/data_plcwrf_t2mean_1982_2019.nc")
pDate, precip, pLong, pLat = reader("Datos/data_plcwrf_pr_1982_2019.nc")
pDate2, precip2, pLong2, pLat2 = reader("Datos/data_plcwrf_pr_1982_2019_masked.nc")

# reading the data from era5 file (from 1982-01-05 to 2019-12-31)
time, longitude, latitude, level, z, q, t, u, v = reader("Datos/data_era5ann_1982_2019.nc", 
                                                         era5=True)

times = [time, time.index('20100101'), time.index('20191231')]

era5data = [z, q, t, u, v]



# TEMPERATURE
tempParams = [500, 128, 5]

# temperature data (in Kelvin)
tempData, tempStore, tempTime = extractData(era5data, tMean, times, "Temperature")

# temperature results
start2 = timeMod.time()
tempPred, tempMetricas = execution(temperatureModel, gaussianLoss, tempData, 
                                  tempStore, tempTime, tempParams)

fin2 = timeMod.time()
timecalc(start2, fin2, f'Tiempo para el modelo de temperatura ({tempParams[0]} epochs, batch size de {tempParams[1]} y {tempParams[2]} repeticiones)')

# temperature boxplot
tCols = ['Mean Bias', 'P2 Bias', 'P98 Bias', 'R (Pearson)', 'std Ratio', 'RMSE', 'Bias WAMS', 'Bias CAMS']

boxplots(tempMetricas, tCols, 
         ['Temperature', f'./Resultados/Temperatura/plots/{tempParams[0]}_epochsTempMetrics.pdf'])

# temperature animation map (over 2017)
animation(tempPred - 273.15, [np.min(tLong), np.max(tLong), np.min(tLat), np.max(tLat)], 
          't', tempTime[1][2546:2920], 365)

# temperature metrics maps
for i in range(len(tCols)):
    mapeo(tempMetricas, [np.shape(tempMetricas)[0], 68, 158], 
      [np.min(tLong), np.max(tLong), np.min(tLat), np.max(tLat)], 
      [get_colormap('tm'), f'./Resultados/Temperatura/plots/tempMetricsMap{i}.pdf'], 
      [0, i])



# PRECIPITATION
precipParams = [100, 16, 4]

# precipitation data (without and with a sea mask respectively)
precipData, precipStore, precipTime = extractData(era5data, precip - 1, times, "Precipitation")
precipData2 = [precipData[0], extractMask(precip2[:times[1]]), precipData[2], extractMask(precip2[times[1]:])]

# precipitation results (using only values over the islands)
start4 = timeMod.time()
precipPred, precipMetricas = execution(precipModel, bernouilliGammaLoss, 
                                       precipData2, precipStore, precipTime, 
                                       precipParams)

fin4 = timeMod.time()
timecalc(start4, fin4, f'Tiempo para el modelo de precipitaciones ({precipParams[0]} epochs, batch size de {precipParams[1]} y {precipParams[2]} repeticiones)')

# fill the complete grid with predicted values for islands
precipData3 = np.reshape(precipData[3], (np.shape(precipData[3])[0], 68*158))
reMask = np.reshape(precip2[times[1]:] >= 0, (np.shape(precip2[times[1]:] >= 0)[0], 68*158))

printeo('Construyendo resultados...')

precipPredValues = unmask(precipPred, np.where(reMask == False, 0, -999.))
precipMetricsValues = unmask(precipMetricas, np.where(reMask[:6, :] == False, 0, -999.))

# precipitation boxplots
pCols = ['Mean Bias', 'P98 Bias', 'R (Spearman)', 'RMSE (Wet days)', 'Bias WetAMS', 'Bias_DryAMS']

boxplots(precipMetricas[:2, :], pCols[:2], 
         ['biases', f'./Resultados/Precipitacion/plots/(mask){precipParams[0]}_epochsPrecipMetrics1.pdf'])

boxplots(precipMetricas[2:4, :], pCols[2:4], 
         ['Precipitation', f'./Resultados/Precipitacion/plots/(mask){precipParams[0]}_epochsPrecipMetrics2.pdf'])

boxplots(precipMetricas[4:, :], pCols[4:], 
         ['Precipitation', f'./Resultados/Precipitacion/plots/(mask){precipParams[0]}_epochsPrecipMetrics3.pdf'])

printeo('Sacando fotogramas...')

# precipitation animation map (over 2017)
animation(precipPredValues, [np.min(pLong), np.max(pLong), np.min(pLat), np.max(pLat)], 
          'p', precipTime[1][2546:2920], 365)

# precipitation metrics maps
for i in range(len(pCols)):
    mapeo(precipMetricsValues, [np.shape(precipMetricsValues)[0], 68, 158], 
      [np.min(pLong), np.max(pLong), np.min(pLat), np.max(pLat)], 
      [get_colormap('pm'), f'./Resultados/Precipitacion/plots/(mask)precipMetricsMap{i}.pdf'], 
      [0, i], True)
