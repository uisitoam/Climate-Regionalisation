import numpy as np 
import time as timeMod
#from bot import bot_texter
from reader import reader
from datafunctions import extractData
from trainfunctions import execution
from nets import temperatureModel, precipModel, gaussianLoss, bernouilliGammaLoss
from plots import boxplots

#start1 = timeMod.time()

# reading the data files 
dateTFN, tMeanTFN, tMaxTFN, tMinTFN, precipTFN = reader("Datos/TFN.dat")
dateTFS, tMeanTFS, tMaxTFS, tMinTFS, precipTFS = reader("Datos/TFS.dat")
dateIZ, tMeanIZ, tMaxIZ, tMinIZ, precipIZ = reader("Datos/Izana.dat")
dateSC, tMeanSC, tMaxSC, tMinSC, precipSC = reader("Datos/StaCruz.dat")

# reading the data from era5 file
time, longitude, latitude, level, z, q, t, u, v = reader("Datos/era5_all_2deg_12h.nc", 
                                                         era5=True)

#fin1 = timeMod.time()

#minutes1, seconds1 = divmod(fin1-start1, 60)
#hours1, minutes1 = divmod(minutes1, 60)

#bot_texter("Tiempo en leer los archivos: %d:%02d:%02d" % (hours1, minutes1, seconds1))

times = [time, time.index('20100101'), time.index('20191231')]

era5data = [z, q, t, u, v]

#start2 = timeMod.time()

#temperature data (in Kelvin)
tempParams = [60, 128, 6]

tMean = [tMeanTFN + 273.15, tMeanTFS + 273.15, tMeanIZ + 273.15, tMeanSC + 273.15] 

tempData, tempStore, tempTime = extractData(era5data, tMean, times, 220)

tempPred, tempMetricas = execution(temperatureModel, gaussianLoss, tempData, 
                                  tempStore, tempTime, tempParams)

#fin2 = timeMod.time()

#minutes2, seconds2 = divmod(fin2-start2, 60)
#hours2, minutes2 = divmod(minutes2, 60)

#bot_texter("Tiempo para el modelo de temperatura (50 epochs, batch size de 128 y 8 repeticiones): %d:%02d:%02d" % (hours2, minutes2, seconds2))

#start3 = timeMod.time()

#precipitation data 
precipParams = [130, 128, 6]

precip = [precipTFN - 1, precipTFS - 1, precipIZ - 1, precipSC - 1]

precipData, precipStore, precipTime = extractData(era5data, precip, times, -1.1)

precipPred, precipMetricas = execution(precipModel, bernouilliGammaLoss, 
                                       precipData, precipStore, precipTime, 
                                       precipParams)



#fin3 = timeMod.time()

#minutes3, seconds3 = divmod(fin3-start3, 60)
#hours3, minutes3 = divmod(minutes3, 60)

#bot_texter("Tiempo para el modelo de precipitaciones determinista (120 epochs, batch size de 128 y 8 repeticiones): %d:%02d:%02d" % (hours3, minutes3, seconds3))

# used metrics
tCols = ['Mean Bias', 'P2 Bias', 'P98 Bias', 'R (Pearson)', 'std Ratio', 'RMSE', 'Bias WAMS', 'Bias CAMS']
pCols = ['Mean Bias', 'P98 Bias', 'R (Spearman)', 'RMSE', 'Bias WetAMS', 'Bias_DryAMS']

boxplots(tempMetricas, tCols, 
         ['Temperature', f'./Resultados/Temperatura/plots/{tempParams[0]}_epochsTempMetrics.pdf'])

boxplots(precipMetricas, pCols, 
         ['Precipitation', f'./Resultados/Precipitacion/plots/{precipParams[0]}_epochsPrecipMetrics.pdf'])


