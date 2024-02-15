import numpy as np
from bot import bot_texter
from reader import reader
from datafunctions import extractData
from trainfunctions import execution
from nets import temperatureModel, precipModel, gaussianLoss, bernouilliGammaLoss
import matplotlib.pyplot as plt


# reading the data files 
dateTFN, tMeanTFN, tMaxTFN, tMinTFN, precipTFN = reader("Datos/TFN.dat")
dateTFS, tMeanTFS, tMaxTFS, tMinTFS, precipTFS = reader("Datos/TFS.dat")
dateIZ, tMeanIZ, tMaxIZ, tMinIZ, precipIZ = reader("Datos/Izana.dat")
dateSC, tMeanSC, tMaxSC, tMinSC, precipSC = reader("Datos/StaCruz.dat")

# reading the data from era5 file
time, longitude, latitude, level, z, q, t, u, v = reader("Datos/era5_all_2deg_12h.nc", 
                                                         era5=True)


#time1 = np.argmax(time == 20100101)
time1 = time.index('20100101')
#time2 = np.argmax(time == 20191231) 
time2 = time.index('20191231')
times = [time, time1, time2]

era5data = [z, q, t, u, v]


#temperature data (in Kelvin)
tMean = [tMeanTFN + 273.15, tMeanTFS + 273.15, tMeanIZ + 273.15, tMeanSC + 273.15] 

tempData, tempStore, tempTime = extractData(era5data, tMean, times, 220)

tempPed, tempMetricas = execution(temperatureModel, gaussianLoss, tempData, 
                                  tempStore, tempTime, [3, 60, 64, 5], False)


#precipitation data 
precip = [precipTFN - 1, precipTFS - 1, precipIZ - 1, precipSC - 1]

precipData, precipStore, precipTime = extractData(era5data, precip, times, -1.1)

precipPred, precipMetricas = execution(precipModel, bernouilliGammaLoss, 
                                       precipData, precipStore, precipTime, 
                                       [3, 100, 64, 5], False)


precipPred_stoc, precipMetricas_stoc = execution(precipModel, bernouilliGammaLoss, 
                                       precipData, precipStore, precipTime, 
                                       [3, 100, 64, 5], True)



tCols = ['Mean Bias', 'P2 Bias', 'P98 Bias', 'R (Pearson)', 'std Ratio', 'RMSE', 'Bias WAMS', 'Bias CAMS']
pCols = ['Mean Bias', 'P98 Bias', 'Spearman coefficient', 'RMSE', 'Bias WetAMS', 'Bias_DryAMS']


## BOXPLOT
#se pinta el bias medio de todo ???? (media de tfn, iz, sc)


# Creating dataset

data = [element for element in tempMetricas]
data2 = [element for element in precipMetricas]
data3 = [element for element in precipMetricas_stoc]
 
fig, ax = plt.subplots(1, 3, figsize=(9,5))
 
# Creating plot
bp = ax[0].boxplot(data, labels=tCols)
bp2 = ax[1].boxplot(data2, labels=pCols)
bp2 = ax[2].boxplot(data3, labels=pCols)

ax[0].set(title='Temperture')
ax[0].set_xticklabels(tCols, rotation=45)
ax[1].set_xticklabels(pCols, rotation=45)
ax[1].set(title='Precipitation(det)')
ax[2].set_xticklabels(pCols, rotation=45)
ax[2].set(title='Precipitation(stoc)')

plt.subplots_adjust(top=0.92, bottom=0.26, left=0.06, right=0.96, hspace=0.2, 
                    wspace=0.2)

fig.savefig('./Resultados/Metrics.pdf')

 
# show plot
plt.show()



