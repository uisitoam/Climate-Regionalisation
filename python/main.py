import numpy as np 
import time as timeMod
from bot import timecalc
from reader import reader
from datafunctions import extractData
from trainfunctions import execution
from nets import temperatureModel, precipModel, gaussianLoss, bernouilliGammaLoss
from plots import boxplots, get_colormap, mapeo, animation


start1 = timeMod.time()

# reading the data files (from 1982-01-05 to 2019-12-31)
tDate, tMean, tLong, tLat = reader("Datos/data_plcwrf_t2mean_1982_2019.nc")
pDate, precip, pLong, pLat = reader("Datos/data_plcwrf_pr_1982_2019.nc")

# reading the data from era5 file (from 1982-01-05 to 2019-12-31)
time, longitude, latitude, level, z, q, t, u, v = reader("Datos/data_era5ann_1982_2019.nc", 
                                                         era5=True)

fin1 = timeMod.time()
timecalc(start1, fin1, 'Tiempo en leer los archivos', True)


times = [time, time.index('20100101'), time.index('20191231')]

era5data = [z, q, t, u, v]





start2 = timeMod.time()

# temperature data (in Kelvin)
tempParams = [60, 128, 6]
tempData, tempStore, tempTime = extractData(era5data, tMean, times, "Temperature")

# temperature results
tempPred, tempMetricas = execution(temperatureModel, gaussianLoss, tempData, 
                                  tempStore, tempTime, tempParams)

fin2 = timeMod.time()
timecalc(start2, fin2, 'Tiempo para el modelo de temperatura (60 epochs, batch size de 128 y 6 repeticiones)', True)

# temperature boxplot
tCols = ['Mean Bias', 'P2 Bias', 'P98 Bias', 'R (Pearson)', 'std Ratio', 'RMSE', 'Bias WAMS', 'Bias CAMS']

boxplots(tempMetricas, tCols, 
         ['Temperature', f'./Resultados/Temperatura/plots/{tempParams[0]}_epochsTempMetrics.pdf'])

# temperature map and animation
mapeo(tempPred - 273.15, [np.shape(tempPred)[0], 68, 158], 
      [np.min(tLong), np.max(tLong), np.min(tLat), np.max(tLat)], 
      [get_colormap('t'), './Resultados/Temperatura/plots/temperatureMap.pdf'], 
      [tempTime[1], '20100307'])

animation(tempPred - 273.15, [np.min(tLong), np.max(tLong), np.min(tLat), np.max(tLat)], 
          't', tempTime[1], 365)

# temperature metrics maps
for i in range(len(tCols)):
    mapeo(tempMetricas, [np.shape(tempMetricas)[0], 68, 158], 
      [np.min(tLong), np.max(tLong), np.min(tLat), np.max(tLat)], 
      [get_colormap('tm'), f'./Resultados/Temperatura/plots/tempMetricsMap{i}.pdf'], 
      [0, i])
  



    
start3 = timeMod.time()

#precipitation data 
precipParams = [120, 128, 6]
precipData, precipStore, precipTime = extractData(era5data, precip - 1, times, "Precipitation")

# precipitation results
precipPred, precipMetricas = execution(precipModel, bernouilliGammaLoss, 
                                       precipData, precipStore, precipTime, 
                                       precipParams)

fin3 = timeMod.time()
timecalc(start3, fin3, 'Tiempo para el modelo de precipitaciones determinista (120 epochs, batch size de 128 y 6 repeticiones)', True)

# precipitation boxplot
pCols = ['Mean Bias', 'P98 Bias', 'R (Spearman)', 'RMSE (Wet days)', 'Bias WetAMS', 'Bias_DryAMS']

boxplots(precipMetricas, pCols, 
         ['Precipitation', f'./Resultados/Precipitacion/plots/{precipParams[0]}_epochsPrecipMetrics.pdf'])

# temperature map and animation
mapeo(precipPred, [np.shape(precipPred)[0], 68, 158], 
      [np.min(pLong), np.max(pLong), np.min(pLat), np.max(pLat)], 
      [get_colormap('p'), './Resultados/Precipitacion/plots/precipitationMap.pdf'], 
      [precipTime[1], '20120307'])

animation(precipPred, [np.min(pLong), np.max(pLong), np.min(pLat), np.max(pLat)], 
          'p', precipTime[1], 365)

# temperature metrics maps
for i in range(len(pCols)):
    mapeo(precipMetricas, [np.shape(precipMetricas)[0], 68, 158], 
      [np.min(tLong), np.max(tLong), np.min(tLat), np.max(tLat)], 
      [get_colormap('pm'), f'./Resultados/Precipitacion/plots/precipMetricsMap{i}.pdf'], 
      [0, i])
