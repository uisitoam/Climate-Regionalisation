import numpy as np
from bot import printeo
from scipy.stats import spearmanr, gamma
import tensorflow as tf
from tqdm import tqdm





def normalize(data, store, name, reshape=False, params=[0, 1]):
    """
    Normalize data using the minimum and maximum values of the data.

    Parameters
    ----------
    data : np.ndarray
        A NumPy array representing the input data.
    store : dict
        A dictionary to store the minimum and maximum values of the data.
    name : str
        A string representing the name of the data.
    reshape : bool, optional    
        A boolean indicating whether to reshape the data. The default is False.
    params : list, optional
        A list of two elements representing the new minimum and maximum values
        to normalize data. The default is [0, 1].

    Returns
    -------
    data : np.ndarray
        A NumPy array representing the normalized data.

    Raises
    ------
    TypeError
        If the input parameters are not of the expected type.
    ValueError
        If the input parameters have inappropriate values or if the minimum and
        maximum values are identical.

    """

    # check input parameters
    assert isinstance(data, np.ndarray), "data must be a numpy array"
    assert isinstance(store, dict), "store must be a dictionary"
    assert isinstance(name, str), "name must be a string"
    assert isinstance(reshape, bool), "reshape must be a boolean"
    assert isinstance(params, list) and len(params) == 2, "params must be a list of two elements"
    

    # reshape samples to (fechas, 10, 17, levels)
    data = np.moveaxis(data, 1, -1) if reshape else data

    try:
        min_val = np.min(data)
        max_val = np.max(data)

        if min_val == max_val:
            raise ValueError("Cannot interpolate with identical minimum and maximum values")
        
        # store min and max values
        if name not in store:
            store[name] = [min_val, max_val] 
            
        data = np.interp(data, (store[name][0], store[name][1]), 
                         (params[0], params[1]))
    
    except ZeroDivisionError:
        raise ValueError("Division by zero during interpolation. Check out minimum and maximum values")
    
    return data





def preprocess_data(samples, labels, mascara, var, store={}, interpol=normalize):
    """
    Preprocess the input data and labels for training the models.

    Parameters
    ----------
    samples : list
        A list of NumPy arrays representing the input data.
    labels : np.ndarray
        A NumPy array representing the labels.
    mascara : np.ndarray
        A NumPy array representing the mask.
    var : str
        A string representing the variable to be used.
    store : dict, optional
        A dictionary to store the minimum and maximum values of the data.
        The default is {}.
    interpol : callable, optional
        A function to interpolate the data. The default is normalize.

    Returns
    -------
    entreno : np.ndarray
        A NumPy array representing the input data.
    labels : np.ndarray
        A NumPy array representing the labels.
    store : dict
        A dictionary containing the minimum and maximum values of the data.

    Raises
    ------
    AssertionError
        If the input parameters do not meet the expected conditions.
        - samples must be a list of 5 arrays with the same shape.
        - labels must be a three-dimensional array with the first dimension
          equal to the first dimension of the arrays that form samples.
        - mascara must be a one-dimensional array.
        - var must be a string equal to 'temp' or 'pr'.
        - store must be a dictionary.
        - interpol must be a function or None.

    """

    # check input parameters
    assert isinstance(samples, list) and len(samples) == 5 and all(isinstance(s, np.ndarray) and samples[0].ndim == 4 and s.shape == samples[0].shape for s in samples), "samples must be a list of 5 4D-arrays with the same shape"
    assert isinstance(labels, np.ndarray) and labels.ndim == 3 and labels.shape[0] == samples[0].shape[0], "labels must be a 3D array, whose first dimension is equal to the first dimension of the arrays that form samples"
    assert isinstance(mascara, np.ndarray) and mascara.ndim == 1, "mascara must be a 1D array"
    assert var in ['temp', 'pr'], "var must be a string equal to 'temp' or 'pr'"
    assert isinstance(store, dict), "store must be a dictionary"
    assert interpol is None or callable(interpol), "if interpol is not None, it must be a function"

    
    interpolador = normalize if interpol is None else interpol

    sampleData = [0] * len(samples)
    nombres = ['z', 'q', 't', 'u', 'v']

    # Normalize each variable of the data
    for i, values in enumerate(zip(samples, nombres)): 
        data, name = values
        sampleData[i] = interpolador(data, store, name, reshape=True)
    
    entreno = np.concatenate((sampleData), axis=3) # make the predictor (5, 10, 17, 3) -> (10, 17, 15)

    labels = np.reshape(labels, (labels.shape[0], -1)) # reshape labels to the shape of the net's output
    labels = labels[:, mascara] # apply the mask to have only the relevant data (islands)

    # normalize just the temperature (if interpol is given)
    if var == 'temp' and interpol is not None:
        labels = interpol(labels, store, 'labels(t)', reshape=False)
    
    else: 
        labels = labels - 1 # 1 is the threshold for precipitation
        labels[labels < 0] = 0  # make sure there are no negative values (bad values have been previously deleted)

    return entreno, labels, store





def computeRainfall(predResults, simulate = False):
    """
    Compute the amount of rainfall from the predicted results.

    Parameters
    ----------
    predResults : np.ndarray
        A NumPy array representing the predicted results.
    simulate : bool, optional
        A boolean indicating whether to obtain stochastic or deterministic 
        results. 
        The default is False (stochastic).

    Returns
    -------
    amountData : np.ndarray
        A NumPy array representing the amount of rainfall.
    prob : np.ndarray
        A NumPy array representing the probability of rainfall.

    Raises
    ------
    AssertionError
        If the input parameters do not meet the expected conditions.
        - predResults must be a 2D array with number of columns divisible by 3.
        - simulate must be a boolean.
        - If simulate is True and the TensorFlow module is not available.

    """

    # check input parameters
    assert isinstance(predResults, np.ndarray) and predResults.ndim == 2 and predResults.shape[1] % 3 == 0, "predResults must be a 2D array with number of columns divisible by 3"
    assert isinstance(simulate, bool), "simulate must be a boolean"

    if simulate and not hasattr(tf, "exp"):
        raise ValueError("TensorFlow module is required")


    # split the predicted results into the probability of rainfall and the log of the alpha and beta parameters
    d = int(predResults.shape[1]/3)
    prob, log_alpha, log_beta = predResults[:, :d], predResults[:, d:2*d], predResults[:, 2*d:]
    
    # stochastic amount of rainfall
    if simulate:
        aux = np.zeros((log_alpha.shape[0], log_alpha.shape[1]))
        alpha = tf.exp(log_alpha)
        beta = tf.exp(log_beta)
        
        for i in range(log_alpha.shape[1]):  
            aux[:, i] = gamma.rvs(alpha[:, i], beta[:, i], size=log_alpha.shape[0])
        
        amountData = aux + 1
    
    # deterministic amount of rainfall
    else:
        amountData = np.array(tf.exp(log_alpha)*tf.exp(log_beta)) + 1

    return amountData, prob





def ams(arr, times, cota=None):
    """
    Calculate the Annual Maximum Spell (AMS) of a given array. 
    - WAMS (Warm Annual Maximum Spell) is the maximum number of consecutive days 
    with temperatures above the 90th percentile.
    - CAMS (Cold Annual Maximum Spell) is the maximum number of consecutive days 
    with temperatures below the 10th percentile.
    - Wet AMS (Wet Annual Maximum Spell) is the maximum number of consecutive days 
    with precipitation above 1 mm.
    - Dry AMS (Dry Annual Maximum Spell) is the maximum number of consecutive days 
    with precipitation below 1 mm.

    Parameters
    ----------
    arr : np.ndarray
        A NumPy array representing the input data.
    times : list
        A list of strings representing the times of the input data.
    cota : float, optional
        A float representing the percentile value or a threshold. The default is None.

    Returns
    -------
    float
        The Annual Maximum Spell (AMS) of the input data.

    Raises
    ------
    AssertionError
        If the input parameters do not meet the expected conditions.
        - arr must be a one-dimensional array.
        - times must be a list of strings of the same length as arr.
        - cota must be a number between 0 and 100 (both not included).

    """

    # check input parameters
    assert isinstance(arr, np.ndarray) and arr.ndim == 1, "arr must be a one-dimensional array"
    assert isinstance(times, list) and all(isinstance(t, str) for t in times) and len(times) == len(arr), "times must be a list of strings of the same length as arr"
    assert cota is None or (isinstance(cota, (int, float)) and 0 < cota < 100), "cota, if not None, must be a number between 0 and 100 (both not included)"
        
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

    # calculate the maximum consecutive length for each year (with a fixed percentile or threshold)
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





def coeficientes(predictions, labels, times, var, annual_max_spell=ams):
    """
    Compute some VALUE COST metrics.
    - For temeperature: Mean Bias, 2nd and 98th percentiles Bias, Pearson correlation 
    coefficient, Standard Deviation Ratio, RMSE, WAMS Bias and CAMS Bias.
    - For precipitation: Mean Bias, 98th percentile Bias, Spearman correlation
    coefficient, RMSE (only for wet days), Wet AMS Bias and Dry AMS Bias.

    Parameters
    ----------
    predictions : np.ndarray
        A NumPy array representing the predicted results.
    labels : np.ndarray
        A NumPy array representing the labels.
    times : list
        A list of strings representing the times of the input data.
    var : str
        A string representing the variable to be used.
    annual_max_spell : callable, optional
        A function to compute the Annual Maximum Spell (AMS). The default is ams.

    Returns
    -------
    np.ndarray
        A NumPy array representing the computed metrics.

    Raises
    ------
    AssertionError
        If the input parameters do not meet the expected conditions.
        - predictions and labels must be arrays of the same shape.
        - times must be a list of strings whose length matches the first dimension of predictions and labels.
        - var must be a string equal to 'temp' or 'pr'.
        - annual_max_spell must be a function.
 
    """

    # check input parameters
    assert isinstance(predictions, np.ndarray) and isinstance(labels, np.ndarray) and predictions.shape == labels.shape, "predictions and labels must be arrays of the same shape"
    assert isinstance(times, list) and all(isinstance(t, str) for t in times) and len(times) == predictions.shape[0], "times must be a list of strings whose length matches dimension 0 of predictions and labels"
    assert var in ['temp', 'pr'], "var must be a string equal to 'temp' or 'pr'"
    assert callable(annual_max_spell), "annual_max_spell must be a function"


    corr = np.zeros(labels.shape[1])
    bias_W = np.zeros(labels.shape[1])
    bias_C = np.zeros(labels.shape[1])

    if var == 'temp':
        biasMean = np.mean(predictions, axis=0) - np.mean(labels, axis=0)
        biasP2 = np.percentile(predictions, 2, axis=0) - np.percentile(labels, 2, axis=0)
        biasP98 = np.percentile(predictions, 98, axis=0) - np.percentile(labels, 98, axis=0)
        std_predictions = np.std(predictions, axis=0)/np.std(labels, axis=0)
        rmse = np.sqrt(np.mean((predictions - labels)**2, axis=0))
        
        printeo('Calculando metricas...')

        for i in tqdm(range(labels.shape[1])): 
            corr[i] = np.corrcoef(predictions[:, i].flatten(), labels[:, i].flatten())[0,1] # pearson
            
            # AMS
            bias_W[i] = int(annual_max_spell(predictions[:, i], times, 90) - 
                                annual_max_spell(labels[:, i], times, 90))
            
            bias_C[i] = int(annual_max_spell(predictions[:, i], times, 10) - 
                                annual_max_spell(labels[:, i], times, 10))
        
        return np.array([biasMean, biasP2, biasP98, corr, std_predictions, rmse, bias_W, bias_C])

    if var == 'pr':
        biasMean= ((np.mean(predictions, axis=0) - np.mean(labels, axis=0))/np.mean(labels, axis=0))*100
        biasP98 = ((np.percentile(np.array(predictions), 98, axis=0) - np.percentile(labels, 98, axis=0))/np.percentile(labels, 98, axis=0))*100
        predictions2, labels2 = predictions, labels
        predictions2[predictions2 < 1] = 0
        labels2[labels2 < 1] = 0
        rmse = np.sqrt(np.mean((predictions2 - labels2)**2, axis=0)) # only wet days

        printeo('Calculando metricas...')

        for i in tqdm(range(labels.shape[1])): 
            corr[i] = spearmanr(predictions[:, i], labels[:, i])[0] # spearman
            
            # AMS
            bias_W[i] = int(annual_max_spell(predictions[:, i], times, 1) - 
                                 annual_max_spell(labels[:, i], times, 1))
            
            bias_C[i] = int(annual_max_spell(predictions[:, i], times) - 
                                 annual_max_spell(labels[:, i], times))
        
        return np.array([biasMean, biasP98, corr, rmse, bias_W, bias_C])





def reMap(values, mask):
    """
    Map the values to the full grid using a mask.

    Parameters
    ----------
    values : np.ndarray
        A NumPy array representing the values to be mapped.
    mask : np.ndarray
        A NumPy array representing the mask of the full region.
    
    Returns
    -------
    fullGrid : np.ndarray
        A NumPy array representing the full region to plot.

    Raises
    ------
    AssertionError
        If the input parameters do not meet the expected conditions.
        - values must be a 2D NumPy array.
        - mask must be a 1D NumPy array of size 10744.

    """

    # check input parameters
    assert isinstance(values, np.ndarray) and values.ndim == 2, "values must be a 2D NumPy array."
    assert isinstance(mask, np.ndarray) and mask.ndim == 1 and mask.size == 68*158, "mask must be a 1D NumPy array of size 10744."

    
    fullGrid = np.zeros((np.shape(values)[0], 68, 158))

    # to avoid nested loops
    flat_indices = np.flatnonzero(mask)
    row_indices, col_indices = np.unravel_index(flat_indices, (68, 158))

    for i in range(np.shape(values)[0]):
        try:
            fullGrid[i, row_indices, col_indices] = values[i, :]
        except IndexError:
            raise ValueError("Not enough elements in values to replace in fullGrid")
    
    return fullGrid
