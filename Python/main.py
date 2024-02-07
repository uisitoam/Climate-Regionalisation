import numpy as np
from bot import bot_texter
from reader import reader
from datathings2 import extractData, execution
from nets import temperatureModel, precModel, gaussianLoss, bernouilliGammaLoss
import matplotlib.pyplot as plt
from ploteo import mapeo 


# reading the data files 
dateTFN, tMeanTFN, tMaxTFN, tMinTFN, precipTFN = reader("Datos/TFN.dat")
dateTFS, tMeanTFS, tMaxTFS, tMinTFS, precipTFS = reader("Datos/TFS.dat")
dateIZ, tMeanIZ, tMaxIZ, tMinIZ, precipIZ = reader("Datos/Izana.dat")
dateSC, tMeanSC, tMaxSC, tMinSC, precipSC = reader("Datos/StaCruz.dat")

# reading the data from era5 file
time, longitude, latitude, level, z, q, t, u, v = reader("Datos/era5_all_2deg_12h.nc", era5=True)


time1 = np.argmax(time == 20100101)
time2 = np.argmax(time == 20191231) 
times = [time1, time2]

era5data = [z, q, t, u, v]

#temperature data (in Kelvin)
tMean = [tMeanTFN + 273.15, tMeanTFS + 273.15, tMeanIZ + 273.15, tMeanSC + 273.15] 

temp_data, temp_store = extractData(era5data, tMean, times, 220)

temp_df, metricas = execution(temperatureModel, gaussianLoss, temp_data, temp_store, [3, 5, 100, 3])

#precipitation data 
precip = [precipTFN - 1, precipTFS - 1, precipIZ - 1, precipSC - 1]















#temp_trainData, temp_trainLabels, temp_testData, temp_testLabels, temp_store = extractData(era5data, tMean, time1, time2, 273.15, 220)
#temp_df, tp, pp = execution(temperatureModel, temp_data, gaussianLoss, temp_store, 3, tCols, filas, eps, reps)
#precip_trainData, precip_trainLabels, precip_testData, precip_testLabels, precip_store = extractData(era5data, precip, times, 0)
#precip_data = [precip_trainData, precip_trainLabels, precip_testData, precip_testLabels]
#precip_df, tp2, pp2 = execution(precModel, precip_data, bernouilliGammaLoss, precip_store, [3, 5, 100, 1])#la tabla lleva otros coeficientes, es pa ver si el modelo ejecuta o no



tCols = ['Mean Bias', 'P2 Bias', 'P98 Bias', 'R (Pearson)', 'std Ratio', 'RMSE', 'Bias WAMS', 'Bias CAMS']
pCols = ['Mean Bias', 'P98 Bias', 'Spearman coefficient', 'RMSE', 'Bias WetAMS', 'Bias_DryAMS']



## BOXPLOT
"""
#se pinta el bias medio de todo ???? (media de tfn, iz, sc)
"""

# Creating dataset
 
data = [element for element in metricas]
#data2 = [element for element in pp2]
 
fig, ax = plt.subplots(1, 2, figsize=(9,5))
 
# Creating plot
bp = ax[0].boxplot(data, labels=tCols)
#bp2 = ax[1].boxplot(data2, labels=pCols)

 
# show plot
plt.show()



