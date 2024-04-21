import numpy as np
from bot import printeo
from scipy.stats import spearmanr, gamma
import tensorflow as tf
from tqdm import tqdm


def interpolNsplit(indexes, info, name, samples=None, labels=None):
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
        
    
    if samples is not None:
        samples2 = np.moveaxis(samples[:indexes[1]+1, :, :, :], 1, -1)
        
        try:
            if np.min(samples2) == np.max(samples2):
                raise ValueError("Cannot interpolate with identical minimum and maximum values")
                
            info[2][name] = [np.min(samples2), np.max(samples2)]
            nSamples = np.interp(samples2, (np.min(samples2), np.max(samples2)), 
                                 (info[0], info[1]))
            
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





def extractData(era5, labs, times, var):
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
    var : str
        'Temperature' or 'Precipitation'
    
    Raises:
    ------
    ValueError:
        - If any parameter has an invalid value.
        - If `times` length is not 3.
    TypeError:
        - If any parameter has an invalid type.
        
    Returns:
    -------
    (data, min_max_values, 
     train_dates, test_dates) : tuple
        - data : list[np.ndarray]
            [entreno, etiquetas, testeo, etiquetasTesteo] : np.ndarray with 
            train a test samples and labels. 
        - minNmax : dict
    """
    
    if not all(isinstance(arr, np.ndarray) for arr in era5):
        raise TypeError("All parameters must be NumPy arrays")
        
    if not isinstance(labs, np.ndarray):
        raise TypeError("labels must be a NumPy array")
        
    if len(times) != 3:
        raise ValueError("times must be a list of three elements")
        

    minNmax = {}
    splitTime = [times[0][:times[1]], times[0][times[1]:]]
    labels = labs[:times[2]+1, :, :]
    
    if var == 'Precipitation':
        labels[labels < 0] = 0
    

    # data from era5 used to train and predict 
    trainData = [0] * len(era5)
    testData = [0] * len(era5)
    nombres = ['z', 'q', 't', 'u', 'v']
    
    for i, values in enumerate(zip(era5, nombres)):
        data, name = values
        trainData[i], testData[i] = interpolNsplit([times[1], times[2]], [0, 1, minNmax], 
                                                   name, data, None)

    #train and test data
    entreno = np.concatenate((trainData), axis=3)
    testeo = np.concatenate((testData), axis=3)
    
    # train and test labels 
    if var == 'Temperature':
        etiquetas, etiquetasTesteo = interpolNsplit([times[1], times[2]], [0, 1, minNmax], 
                                                       'labels(t)', None, labels)
    elif var == 'Precipitation':
        etiquetas, etiquetasTesteo = interpolNsplit([times[1], times[2]], [0, 1, minNmax], 
                                                    'labels(p)', None, labels)
    
    else: 
        raise ValueError('Invalid variable: must be `Temperture` or `Precipitation`')
    
    return [entreno, etiquetas, testeo, etiquetasTesteo], minNmax, splitTime





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
        - If any parameter has an invalid type.

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
        Count annual maximum annual spells for a given year/s
        and obtain the median. 

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
            - If any parameter has an invalid type.
        ValueError
            - If cota is not a positive number between 0 and 100.
        KeyError
            - If the year key doesn't exist yet (first encounter). 
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
    
        for i in tqdm(range(outs)): 
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
        
        printeo('Calculando metricas...')
        
        for i in tqdm(range(outs)):            
            spcorr[i] = spearmanr(np.array(predict[:, i]), labels[:, i])[0] #1d arrays
            mask = predict[:, i] >= 1
            rmse[i] = rootmse(np.array(predict[:, i][mask]), labels[:, i][mask])
            
            bias_WetAMS[i] = int(ams(predict[:, i], times, 1) - 
                                 ams(labels[:, i], times, 1))
            
            bias_DryAMS[i] = int(ams(predict[:, i], times) - 
                                 ams(labels[:, i], times))
        
        return np.array([biasMean, biasP98, spcorr, rmse, bias_WetAMS, bias_DryAMS])
    




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
        Whether to simulate rainfall amounts deterministcly or
        stochastically (default: False, deterministic).
    
    Raises
    ------
    ValueError
        - If `trainData` or `predResults` has invalid shapes.
    TypeError
        - If any parameter has an invalid type.

    Returns
    -------
    amountData : np.ndarray
        Predicted rainfall amounts for each sample and prediction time.
    prob : np.ndarray
        Probability of occurrence
        
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
    d = int(np.shape(predResults)[1]/3)
    prob, log_alpha, log_beta = predResults[:, :d], predResults[:, d:2*d], predResults[:, 2*d:]
    
    if simulate:
        aux = np.zeros((np.shape(log_alpha)[0], np.shape(log_alpha)[1]))
        alpha = tf.exp(log_alpha)
        beta = tf.exp(log_beta)
        
        for i in range(np.shape(log_alpha)[1]):  
            aux[:, i] = gamma.rvs(alpha[:, i], beta[:, i], size=np.shape(log_alpha)[0])
        
        amountData = aux + 1
    
    else:
        amountData = np.array(tf.exp(log_alpha)*tf.exp(log_beta)) + 1

    return amountData, prob





def extractMask(labels):
    """
    Fix to zero precipitation values below 1 on masked arrays. 

    Parameters
    ----------
    labels : np.ndarray
        Masked array to fix its values.

    Returns
    -------
    etiquetas : np.ndarray
        Array with fix values

    """
    etiquetas = np.reshape(labels[labels >= 0], (np.shape(labels)[0], -1)) - 1
    etiquetas[etiquetas < 0] = 0

    return etiquetas





def unmask(values, shapeResults):
    """
    Fill the full grid with predicted values for islands 

    Parameters
    ----------
    values : np.ndarray
        Masked array. Contains values from the islands themself.
    shapeResults : np.ndarray
        Array of zeros with the shape of the full grid. 

    Returns
    -------
    shapeResults : np.ndarray
        Array with the results of the islands placed on each respective 
        positions on the full grid. 

    """
    for i in tqdm(range(np.shape(shapeResults)[0])):
        k = 0
        for j in range(np.shape(shapeResults)[1]):
            if shapeResults[i,j] < -1:
                try:
                    shapeResults[i,j] = values[i,k]

                except ValueError:
                    shapeResults[i,j] = 0

                k += 1
    
    return shapeResults    
    
    