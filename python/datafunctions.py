import numpy as np


def interpolNsplit(indexes, info, name, samples=None, labels=None, masked=None):
    """
    This function splits data into train and test subsets (samples and labels).

    Parameters
    ----------
    indexes : list
        Contains indexes to split the data into train and test subsets.
    info : list
        Contains the information about minimun (first element) and maximum (second
        element) values to interpol to, and the dictionary (third element) with 
        the real max and min values of the data.
    name : str
        Label name to look up the values on the dictionary.
    samples : np.ndarray, optional
        data to train with (or to predict with). To treat samples, keep 
        labels parameter None. The default is None.
    labels : np.ndarray, optional
        Labels of the samples. To treat labels, keep samples parameter None. 
        The default is None.
    masked : np.ndarray, optional
        Mask to apply to samples, following the one applied to the labels (only 
        for temperature, for precipitation, keep it None). The default is None.

    Returns
    -------
    train : np.ndarray
        Train data.
    test : np.ndarray
        Test data.

    """
    
    
    if samples is not None:
        samples2 = samples[:indexes[1], :, :, :]
        samples3 = np.moveaxis(samples2[masked, :, :, :], 1, -1)
        info[2][name] = [np.min(samples3), np.max(samples3)] 
        nSamples = np.interp(samples3, (np.min(samples3), np.max(samples3)), 
                             (info[0], info[1])) 
        train = nSamples[:indexes[0], :, :, :] 
        test = nSamples[indexes[0]:, :, :, :]

        return train, test
    
    if labels is not None:
        info[2][name] = [np.min(labels), np.max(labels)] 
        
        if name == 'labels(t)':
            nLabels = np.interp(labels, (np.min(labels), np.max(labels)), 
                                (info[0], info[1]))
        
        elif name == 'labels(p)':
            nLabels = labels
            
        train = nLabels[:indexes[0]]
        test = nLabels[indexes[0]:]
        
        return train, test





def extractData(era5, labs, times, cota):
    """
    This function treat the data for the network. Delete bad reading, splits 
    and normalized it (when proceeds), and give train and test subsets of 
    samples and labels, as well as the dictionary with original values. 

    Parameters
    ----------
    era5 : list
        Era5 data that will be used for the model to train and predict. It is a 
        list of arrays. 
    labs : list
        Data to be used as label at each station. It is a list of arrays.
    times : list
        Contains the values needed to split data into train (first element) and 
        test subsets (second element), as well as the dates of each data set. 
    cota : int
        Cut off for bad readings.
        - Temperature: unsual reads for TF, like under 220 K.
        - Precipitation: reads under 0 mm ()
 
    Returns
    -------
    entreno : np.ndarray 
        Train data. 
    etiquetas : np.ndarray
        Labels of the train data.
    testeo : np.ndarray
        Test data.
    etiquetasTesteo : np.ndarray
        Labels of the test data. 
    minNmax : dict
        Contains min and max values of all data. 
    timeMasked : list
        Contains the dates of the data in order to calculate some metrics. It 
        is a list of two arrays, the dates for train and test samples.

    """
    
    stTFN = labs[0][:times[2]] 
    stTFS = labs[1][:times[2]] 
    stIZ = labs[2][:times[2]] 
    stSC = labs[3][:times[2]] 
    
    # 220 for temperature, -1.1 for precipitations
    mask = (stTFN >= cota) & (stIZ >= cota) & (stSC >= cota) 
    labels = np.concatenate((stTFN[mask].reshape(-1, 1), stIZ[mask].reshape(-1, 1), 
                             stSC[mask].reshape(-1, 1)), axis=1)
    
    timeCut = times[0][:times[2]]
    timeMasked = [b for a, b in zip(mask, timeCut) if a == True]
    times_Masked = [timeMasked[:times[1]], timeMasked[times[1]:]]
    
    
    
    if cota < 1:
        labels[labels < 0] = 0
        
    minNmax = {}
    
    # data from era5 used to train and predict 
    zTrain_samples, zTest_samples = interpolNsplit([times[1], times[2]], [0, 1, minNmax], 
                                                   'z', era5[0], None, mask)
    
    qTrain_samples, qTest_samples = interpolNsplit([times[1], times[2]], [0, 1, minNmax], 
                                                   'q', era5[1], None, mask)
    
    tTrain_samples, tTest_samples = interpolNsplit([times[1], times[2]], [0, 1, minNmax], 
                                                   't', era5[2], None, mask)
    
    uTrain_samples, uTest_samples = interpolNsplit([times[1], times[2]], [0, 1, minNmax], 
                                                   'u', era5[3], None, mask)
    
    vTrain_samples, vTest_samples = interpolNsplit([times[1], times[2]], [0, 1, minNmax], 
                                                   'v', era5[4], None, mask)

    #train and test data
    entreno = np.concatenate((zTrain_samples, qTrain_samples, tTrain_samples, 
                              uTrain_samples, vTrain_samples), axis=3)
    
    testeo = np.concatenate((zTest_samples, qTest_samples, tTest_samples, 
                             uTest_samples, vTest_samples), axis=3)
    
    # train and test labels 
    if cota > 1:
        etiquetas, etiquetasTesteo = interpolNsplit([times[1], times[2]], [0, 1, minNmax], 
                                                       'labels(t)', None, labels, None)
    elif cota < 1:
        etiquetas, etiquetasTesteo = interpolNsplit([times[1], times[2]], [0, 1, minNmax], 
                                                    'labels(p)', None, labels, None)
    
        
    return [entreno, etiquetas, testeo, etiquetasTesteo], minNmax, times_Masked











    
    
    