import numpy as np
from scipy.stats import spearmanr, gamma
import tensorflow as tf
from bot import bot_texter


def interpolNsplit(indexes, info, name, samples=None, labels=None, masked=None):
    """
    This function splits data into train and test subsets (samples and labels).

    Parameters
    ----------
    indexes : list
        List containing two integer indices to split the data.
    info : list
        List containing three elements:
            - Minimum value for interpolation (first element).
            - Maximum value for interpolation (second element).
            - Dictionary with real minimum and maximum values (third element).
    name : str
        Label name to reference in the `info` dictionary. It is `'labels(t)'` 
        for temperature and `'labels(p)'` for precipitation.
    samples : np.ndarray, optional
        Data to train with (or predict with). Provide `samples` and `masked` 
        if treating samples.
    labels : np.ndarray, optional
        Labels of the samples. Provide only `labels` if treating labels.
    masked : np.ndarray, optional
        Mask to apply to samples, used only with temperature labels 
        (`name == 'labels(t)'`). The default is None.

        
    Raises
    ------
    TypeError:
        - If any parameter has an unexpected type.
    ValueError:
        - If `indexes` does not have two integers or has invalid values.
        - If `info` is not a list of three elements or has invalid structure.
        - If `name` is not a string.
        - If `samples` or `labels` are not NumPy arrays when provided.
        - If `masked` is not a NumPy array when provided.
        - If `samples` and `masked` have different lengths.
        - If minimum and maximum values for interpolation are identical.
        - If division by zero occurs during interpolation.
        - If `name` is not found in the `info` dictionary.

    Returns
    -------
    (train, test): tuple
        - train: Train data.
        - test: Test data.

    """
    
    if not isinstance(indexes, list) or len(indexes) != 2 or not all(isinstance(i, int) for i in indexes):
        raise TypeError("indexes must be a list of two integers")
    
    if indexes[0] > indexes[1]:
        raise TypeError("Second index must be greater than the first one")
    
    if not isinstance(info, list) or len(info) != 3:
        raise TypeError("info must be a list of 3 elements")    
        
    if info[0] > info[1]:
        info[0], info[1] = info[1], info[0]
    
    if not isinstance(info[2], dict):
        raise TypeError("Third element of info must be a dictionary")
    
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    
    if samples is not None and not isinstance(samples, np.ndarray):
        raise TypeError("samples must be a NumPy array")
        
    if labels is not None and not isinstance(labels, np.ndarray):
        raise TypeError("labels must be a NumPy array")
    
    if masked is not None and not isinstance(masked, np.ndarray):
        raise TypeError("masked must be a NumPy array")
        
    
    if samples is not None:
        samples2 = samples[:indexes[1], :, :, :]
        
        if samples is not None and masked is not None and np.shape(samples2)[0] != len(masked):
            raise TypeError("samples (until indexes[1]) and masked must be the same length")
            
        samples3 = np.moveaxis(samples2[masked, :, :, :], 1, -1)
        
        try:
            if np.min(samples3) == np.max(samples3):
                raise ValueError("Cannot interpolate with identical minimum and maximum values")
                
            info[2][name] = [np.min(samples3), np.max(samples3)]
            nSamples = np.interp(samples3, (np.min(samples3), np.max(samples3)), (info[0], info[1]))
            
        except ZeroDivisionError:
            raise ValueError("Division by zero during interpolation. Check out minimum and maximum values")

        train = nSamples[:indexes[0], :, :, :] 
        test = nSamples[indexes[0]:, :, :, :]

        return train, test


    elif labels is not None:
        info[2][name] = [np.min(labels), np.max(labels)] 
        
        if np.min(labels) == np.max(labels):
                raise ValueError("Cannot interpolate with identical minimum and maximum values")
        
        if name == 'labels(t)':
            nLabels = np.interp(labels, (np.min(labels), np.max(labels)), 
                                (info[0], info[1]))
        
        elif name == 'labels(p)':
            nLabels = labels
        
        else:
            raise TypeError("name not found on info dictionary")
            
        train = nLabels[:indexes[0]]
        test = nLabels[indexes[0]:]
        
        return train, test




def extractData(era5, labs, times, cota):
    """
    Preprocesses and splits ERA5 and label data into training and testing sets: 
    delete bad reading, splits and normalized it (when proceeds).

    Parameters:
    ----------
    era5 : list[np.ndarray]
        List of ERA5 data arrays (z, q, t, u, v).
    labs : list[np.ndarray]
        List of label arrays (TFN, TFS, IZ, SC).
    times : list[np.ndarray, int, int]
        List containing three elements:
            - Total number of time steps (dates).
            - Twp indexes to split data into train and test sets.
    cota : int, float
        Cut-off threshold for bad readings:
            - Temperature: values below cota are invalid.
            - Precipitation: negative values are set to 0.
    
    Raises:
    ------
    ValueError:
        - If any parameter has an invalid type or value.
        - If `times` length is not 3.
    TypeError:
        - If any parameter is not a NumPy array.
        
    Returns:
    -------
    (data, min_max_values, 
     train_dates, test_dates) : tuple
        - data : list[np.ndarray]
            [entreno, etiquetas, testeo, etiquetasTesteo] : np.ndarray with 
            train a test samples and labels. 
        - minNmax : dict
        - times_Masked : list[list]
            [timeMasked[:times[1]], timeMasked[times[1]:]] : list of train and 
            test dates.
    """
    
    if not all(isinstance(arr, np.ndarray) for arr in era5 + labs):
        raise TypeError("All parameters must be NumPy arrays")
        
    if len(times) != 3:
        raise ValueError("times must be a list of three elements")
        
    
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
    
    trainData = [0] * len(era5)
    testData = [0] * len(era5)
    nombres = ['z', 'q', 't', 'u', 'v']
    
    for i, values in enumerate(zip(era5, nombres)):
        data, name = values
        trainData[i], testData[i] = interpolNsplit([times[1], times[2]], [0, 1, minNmax], 
                                                   name, data, None, mask)

    #train and test data
    entreno = np.concatenate((trainData), axis=3)
    testeo = np.concatenate((testData), axis=3)
    
    # train and test labels 
    if cota > 1:
        etiquetas, etiquetasTesteo = interpolNsplit([times[1], times[2]], [0, 1, minNmax], 
                                                       'labels(t)', None, labels, None)
    elif cota < 1:
        etiquetas, etiquetasTesteo = interpolNsplit([times[1], times[2]], [0, 1, minNmax], 
                                                    'labels(p)', None, labels, None)
    
    else: 
        raise ValueError('Invalid cota value: must be greater than zero')
    
        
    return [entreno, etiquetas, testeo, etiquetasTesteo], minNmax, times_Masked





def coeficientes(predict, labels, times, var):
    """
    Obtain metrics (coefficients) to evaluate the permorfance of the model. 
    - Temperature: Mean bias, percentile 2 and 98 bias, pearson correlation
    coefficient, standard deviation ratio, root mean square error and bias for 
    warm and cold annual max spells (WAMS and CAMS).
    - Precipitation: Mean bias, percentile 98 bias, spearson correlation 
    coefficient, root mean square error, bias for wet and dry annual max spells
    (WetAMS and DryAMS) and (yet to be implemented) ROC skill score (ROCSS).

    Parameters
    ----------
    predict : np.ndarray
        Predicted data.
    labels : np.ndarray
        True data.
    times : list
        Contains the dates of the data. 
    var : str
        Model we are dealing with, either temperature or precipitation.
    
    Raises
    ------
    ValueError
        If `var` is not "temperature" or "precipitation".
    TypeError
        - If `predict` or `labels` is not a NumPy array.
        - If `times` is not a list of strings.

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
    
    if not isinstance(predict, np.ndarray):
        raise TypeError("predict must be a NumPy array")
        
    if not isinstance(labels, np.ndarray):
        raise TypeError("labels must be a NumPy array")
        
    if np.shape(predict)[0] != np.shape(labels)[0] or np.shape(predict)[1] != np.shape(labels)[1]:
            raise ValueError("predict and labels must have the same shape")
        
    if not isinstance(times, list) or not all(isinstance(i, str) for i in times):
        raise TypeError("times must be a list of strings")
        
    if var not in ("temperature", "precipitation"):
        raise ValueError("var must be 'temperature' or 'precipitation'")


    def rootmse(preds, targets):
        return np.sqrt(((preds - targets) ** 2).mean())

    #max spells 
    def ams(arr, times, cota=None):
        """
        

        Parameters
        ----------
        arr : np.ndarray
            Values to be looking at.
        times : list
            Times of the given values. The dates should be strings.
        cota : (int, float), optional
            Cut off value to . The default is None.

        Raises
        ------
        TypeError
            arr must be a np.ndarray, times must be a list and cota must be a 
            positive number between 0 and 100.
        ValueError
            cota must be a positive number between 0 and 100.
        KeyError
            It will raise if the year key doesn't exist yet (first encounter). 
            An except block catches the KeyError and initializes a new list 
            for the encountered year in the dictionary.

        Returns
        -------
        float
            Median of the annual maximum spells over the years given in times.

        """
            
        if not isinstance(times, list):
            raise TypeError("times must be a list of strings")
            
        if cota is not None and (not isinstance(cota, (int, float)) or cota < 0 or cota > 100):
            raise ValueError("cota must be a positive number between 0 and 100")
            
        if cota is not None and cota == 0:
            cota = None
    
        # group values by year 
        values_by_year = {}
        
        for value, date in zip(arr, times):
            year = int(date[:4])
            
            try:
                values_by_year[year].append(value)
                
            except KeyError:
                values_by_year[year] = [value]
    
        max_consecutive_lengths = np.zeros(len(values_by_year))
        
        for i, dic in enumerate(values_by_year.items()):
            year, values = dic
            max_consecutive = 0
            current_consecutive = 0
            percentile = np.percentile(arr, cota) if cota is not None and cota > 1 else cota
            
            for value in values:
                condition = (value >= percentile if cota is not None and (cota > 50 or cota == 1) 
                             else value <= percentile if cota is not None and cota < 50
                             else value < 1
                             )
                
                if condition:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                    
                else:
                    current_consecutive = 0
                    
            max_consecutive_lengths[i] = max_consecutive
    
        return np.median(max_consecutive_lengths)
    
    outs = np.shape(labels)[1]
    
    if var == 'temperature':
        biasMean = np.mean(predict, axis=0) - np.mean(labels, axis=0)
        biasP2 = np.percentile(predict, 2, axis=0) - np.percentile(labels, 2, axis=0)
        biasP98 = np.percentile(predict, 98, axis=0) - np.percentile(labels, 98, axis=0)
        pcorr = np.zeros(outs)
        std_predictions = np.std(predict, axis=0)/np.std(labels, axis=0)
        rmse = np.zeros(outs)
        bias_WAMS = np.zeros(outs)
        bias_CAMS = np.zeros(outs)
    
        for i in range(outs): 
            pcorr[i] = np.corrcoef(predict[:, i].flatten(), labels[:, i].flatten())[0,1]
            rmse[i] = rootmse(predict[:, i], labels[:, i])
            
            # AMS
            bias_WAMS[i] = int(ams(predict[:, i], times, 90) - 
                               ams(labels[:, i], times, 90))
            
            bias_CAMS[i] = int(ams(predict[:, i], times, 10) - 
                               ams(labels[:, i], times, 10))
        
        return np.array([biasMean, biasP2, biasP98, pcorr, std_predictions, rmse, bias_WAMS, bias_CAMS])
        
    
    if var == 'precipitation':
        biasMean = ((np.mean(predict, axis=0) - 
                     np.mean(labels, axis=0))/np.mean(labels, axis=0))*100
        
        biasP98 = ((np.percentile(np.array(predict), 98, axis=0) - 
                    np.percentile(labels, 98, axis=0))/np.percentile(labels, 98, axis=0))*100
        
        spcorr = np.zeros(outs)
        rmse = np.zeros(outs)
        bias_WetAMS = np.zeros(outs)
        bias_DryAMS = np.zeros(outs)
        
        for i in range(outs):            
            spcorr[i] = spearmanr(np.array(predict[:, i]), labels[:, i])[0] #1d arrays
            rmse[i] = rootmse(np.array(predict[:, i]), labels[:, i])
            
            bias_WetAMS[i] = int(ams(predict[:, i], times, 1) - 
                                 ams(labels[:, i], times, 1))
            
            bias_DryAMS[i] = int(ams(predict[:, i], times) - 
                                 ams(labels[:, i], times))
        
        return np.array([biasMean, biasP98, spcorr, rmse, bias_WetAMS, bias_DryAMS])
    




#simulate = True es stochastic
def computeRainfall(trainData, predResults, simulate = False):
    """
    Calculates predicted rainfall amounts based on input data and a given 
    threshold, which is also calculated.

    Parameters
    ----------
    trainData : list
        Contains training data: samples, labels, and probabilities.
    predResults : np.ndarray
        Predicted probabilities, log-transformed alpha and beta parameters.
    simulate : bool, optional
        Whether to simulate rainfall amounts (default: False).
    
    Raises
    ------
    ValueError
        - If `trainData` or `predResults` has incorrect shapes or data types.
    TypeError
        - If elements in `trainData` or `predResults` are not NumPy arrays.

    Returns
    -------
    np.ndarray
        Predicted rainfall amounts for each sample and prediction time.
        
    """
    
    if not isinstance(trainData, list) or len(trainData) != 3:
        raise ValueError("trainData must be a list of length 3 (samples, labels, probs)")
            
    if not all(isinstance(i, np.ndarray) for i in trainData):
        raise TypeError("Elements in trainData must be NumPy arrays")

    if not isinstance(predResults, np.ndarray):
        raise TypeError("predResults must be a NumPy array")

    if simulate and not hasattr(tf, "exp"):
        raise ValueError("TensorFlow module is required")

    
    trainSamples, trainLabels, trainProb = trainData
    prob, log_alpha, log_beta = predResults[:, :3], predResults[:, 3:6], predResults[:, 6:]
    
    if simulate:
        aux = np.zeros((np.shape(log_alpha)[0], np.shape(log_alpha)[1]))
        alpha = tf.exp(log_alpha)
        beta = tf.exp(log_beta)
        
        for i in range(np.shape(log_alpha)[1]):  
            aux[:, i] = gamma.rvs(alpha[:, i], beta[:, i], size=np.shape(log_alpha)[0])
        
        amountData = aux + 1
    
    else:
        amountData = np.array(tf.exp(log_alpha)*tf.exp(log_beta)) + 1
        
    porcentaje = np.sum(trainLabels <= 0, axis=0) / np.shape(trainLabels)[0]
    cota = np.diag(np.nanquantile(trainProb, porcentaje, axis=0))
    rainAmount = np.zeros_like(amountData)
    
    
    for i in range(np.shape(rainAmount)[1]):
        for j in range(np.shape(rainAmount)[0]):
            if prob[j, i] < cota[i]:
                rainAmount[j, i] = 0
            else:
                rainAmount[j, i] = amountData[j, i]
                
    #rainAmount2 = np.where(prob < cota, 0, amountData) #ver si es exactamente lo mismo
    #rainAmount = rainAmount*prob
        
    return rainAmount





    
    
    