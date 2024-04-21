import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy.feature as cpf
import cartopy.crs as ccrs
from tqdm import tqdm


colores = ['#FF0000', '#FF9B00', '#F5DE00', '#51FF53', '#00A902', '#00FF8F', 
           '#00F5E6', '#00B2FF', '#0032FF', '#B900FF', '#8700FF', '#FF86F6', 
           '#EA00D9', '#B00070', '#A4A4A4']

def learningCurve(ejey, savePath):
    """
    This functions plots the leraning curves of the models. 

    Parameters
    ----------
    ejey : list
        List containing the training and the validation loss as two different 
        NumPy arrays.
    ep : int
        Epochs
    savePath : str
        Path where the plot will be saved.

    Raises
    ------
    ValueError
        If `ejey` do not contain 2 NumPy arrays.
    PermissionError
        Insufficient permission to save the plot on the specified path.
    FileNotFoundError
        If the specified save path does not exist and is not writable.

    Returns
    -------
    None.

    """

    if len(ejey) != 2 or not all(isinstance(i, np.ndarray) for i in ejey):
        raise ValueError("ejey must be a list of 2 NumPy arrays.")

    fig, ax = plt.subplots(1, 1, figsize=(9,5))

    ejex = np.arange(1, len(ejey[0]) + 1)

    ax.plot(ejex, ejey[0], color=colores[5], label='Training Loss')
    ax.plot(ejex, ejey[1], color=colores[9], label='Validation Loss')
    ax.set(xlabel='Epochs', ylabel='Loss', title='Training and Validation Loss')
    ax.legend()
    
    if savePath: # Validate directory or file path
        try:
            with open(savePath, 'x') as _:
                pass  # Create an empty file to indicate write permission
                
        except PermissionError:
            raise PermissionError(f"Insufficient permissions to save the plot to '{savePath}'.")
        
        except FileNotFoundError:
            raise FileNotFoundError(f"Save path '{savePath}' not found.")
        
        fig.savefig(savePath)
    
    return 





def boxplots(metrics, metricLabels, params):
    """
    Creates a boxplot of given metrics.

    Parameters
    ----------
    metrics : np.ndarray
        NumPy array with the metrics being evaluated in that model.
    metricLabels : list
        List containing the names of the previous metrics. 
    params : list
        List of strings with the model being evaluated and the saving path 
        for the plot.

    Raises
    ------
    TypeError
        If the title parameter in `params` is not "Temperature" or "Precipitation".
    ValueError
        If the dimensions of `metrics` and `metricLabels` don't match.
    PermissionError
        Insufficient permissions to save the plot to specified path.
    FileNotFoundError
        If the specified save path does not exist and is not writable.

    Returns
    -------
    Matplotlib figure object

    """
    
    if not isinstance(metrics, np.ndarray):
        raise TypeError("metrics must be a NumPy array")
    
    if not all(isinstance(i, str) for i in metricLabels):
        raise TypeError("metricLabels must be a list of only strings")
    
    if not all(isinstance(i, str) for i in params):
        raise TypeError("params must be a list of only strings")

    if np.shape(metrics)[0] != len(metricLabels):
        raise ValueError("metrics and metricLabels must have the same length.")
   
    if params[0] not in ('Temperature', 'Precipitation'):
        raise TypeError("Title must be `Temperature` or `Precipitation`")
        

    def approx(value, expect):
      resto = value % expect
      
      if value < 0:
        return value - resto
    
      else:
        return value + (expect - resto)
    
    data = [element for element in metrics]
    
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    
    ax.boxplot(data, vert=False, showfliers = False, 
               notch = True, labels=metricLabels)
    
    ax.set(title=params[0])
    ax.set_yticklabels(metricLabels)
    
    q1 = np.nanquantile(metrics, 0.25, axis=1)
    q3 = np.nanquantile(metrics, 0.75, axis=1)
    iqr = q3 - q1
    
    if params[0] == 'Temperature':
        step = 0.5
        xticks = np.arange(approx(np.min(q1 - 1.5*iqr), step) - step, 
                           approx(np.max(q3 + 1.5*iqr), step) + step, step)
        
        xlabels = [f'{x:1.1f}' for x in xticks]
        ax.set_xticks(xticks, minor=True, labels=xlabels)
        ax.tick_params(axis='x', labelsize=10)
     
    elif params[0] == 'Precipitation':
        step = 50
        xticks = np.arange(approx(np.min(q1 - 1.5*iqr), step) - step, 
                           approx(np.max(q3 + 1.5*iqr), step) + step, step)
        
        xlabels = [f'{x:1.1f}' for x in xticks]
        ax.set_xticks(xticks, labels=xlabels, minor=True)
        ax.tick_params(axis='x', which='both', labelsize=8)
    
    
    plt.subplots_adjust(top=0.935, bottom=0.175, left=0.16, right=0.92, 
                        hspace=0.2, wspace=0.2)
    
    try:
        with open(params[1], 'x') as _:
            pass  # Create an empty file to indicate write permission
    
    except PermissionError:
        raise PermissionError(f"Insufficient permissions to save the plot to '{params[1]}'.")
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Save path '{params[1]}' not found.")

    fig.savefig(params[1], bbox_inches='tight')
    
    return 





def get_colormap(var):
    """
    Return a colormap for the given variable.

    Parameters
    ----------
    var : str
        Must be `t` for Temperture and `p` for Precipitation, 
        'tm' for temperature metrics and 'pm' for precipitation metrics.

    Raises
    ------
    TypeError
        If var is not 't', 'tm', 'p' or 'pm'.

    Returns
    -------
    Matplotlib LinearSegmentedColormap object.

    """
    
    if var not in ('t', 'p', 'tm', 'pm'):
        raise TypeError("Variable must be 't' or 'tm' for temperature or 'p' or 'pm' for precipitation.")
    
    colormaps = {'t': [(0.5, 0.0, 0.5), (0.3, 0.0, 0.7), (0.0, 0.0, 0.5),
                       (0.2, 0.0, 1.0), (0.0, 0.0, 1.0), (0.4, 0.4, 1.0), 
                       (0.2, 0.8, 0.8), (0.5, 1.0, 0.5), (0.0, 1.0, 0.0), 
                       (0.3, 1.0, 0.3), (1.0, 1.0, 0.0), (1.0, 0.8, 0.0), 
                       (1.0, 0.6, 0.0), (1.0, 0.5, 0.0), (1.0, 0.3, 0.0), 
                       (0.8, 0.2, 0.0), (0.5, 0.0, 0.0)], 
                 'p': [(1.000, 1.000, 1.000), (0.929, 0.929, 0.964), (0.857, 0.857, 0.929), 
                       (0.786, 0.786, 0.893), (0.714, 0.714, 0.857), (0.643, 0.643, 0.821),
                       (0.571, 0.571, 0.786), (0.500, 0.500, 0.750), (0.429, 0.429, 0.714),
                       (0.357, 0.357, 0.679), (0.286, 0.286, 0.643), (0.214, 0.214, 0.607),
                       (0.143, 0.143, 0.571), (0.071, 0.071, 0.536), (0.000, 0.000, 0.500),
                       (0.000, 0.000, 0.500)],
                 'tm': 'OrRd', 'pm': 'BrBG'}
    if var == 't' or var == 'p':
        return LinearSegmentedColormap.from_list("full_spectrum", colormaps[var])
    
    else: 
        return colormaps[var]

    



def mapeo(data, dataDim, dataExt, params, time, seamask=False, imgExt=None):
    """
    Make a map plot of the given area for the specified variable.

    Parameters
    ----------
    data : np.ndarray
        Data (temperature or rain amount) to be plot over the map.
    dataDim : list
        Dimensions of the given data like (time, latitude, longitude).
    dataExt : list
        Extremal values of latitude and longitude of the data.
    params : list
        Contains the colormap and the save path.
    time : list
        Contains all dates with available data and the exact day to be plotted.
    seamask : bool, optional
        Add a white mask to the sea. The default is False.
    imgExt : list, optional
        Extent of the plotted image, by default is the same as the data 
        extension. The default is None.

    Raises
    ------
    TypeError: 
        If any of the following conditions are met:
            - `data` is not a numpy array.
            - `dataDim` is not a list of length 3.
            - `dataExt` is not a list of length 4.
            - `params` is not a string list of length 2.
            - `colorMap` in `params` is not a matplotlib colormap object.
            - `savePath` in `params` is not a string.
    ValueError
        If the dimensions of `data` and `dataDim` are not compatible.
    PermissionError
        Insufficient permissions to save the plot to specified path.
    FileNotFoundError
        If the specified save path does not exist and is not writable.

    Returns
    -------
    Matplotlib figure object.

    """

    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a NumPy array.")
  
    if not isinstance(dataDim, list) or len(dataDim) != 3 or not all(isinstance(i, int) for i in dataDim):
        raise ValueError("dataDim must be a list of three integers.")
  
    if not isinstance(dataExt, list) or len(dataExt) != 4 or not all(isinstance(i, np.float32) for i in dataExt):
        raise TypeError("dataExt must be a list of 4 floats.")
  
    if not isinstance(params, list) or len(params) != 2:
        raise TypeError("Params must be a tuple of length 2 containing (colormap, save path).")
        
    if not isinstance(time, list) or len(time) != 2:
        raise TypeError("time must be a list of two elements")
    

    colorMap, savePath = params
    data1 = np.reshape(data, (dataDim[0], dataDim[1], dataDim[2]))
    
    if 'Metrics' in savePath:
        data2 = data1[time[1], ::-1, :]
        
    else:
        if not isinstance(time, list) or len(time) != 2:
            raise TypeError("time must be a list containing (Dates, given date)")
          
        if int(time[1]) < int(time[0][0]) or int(time[1]) > int(time[0][-1]):
            raise TypeError(f'Date must be contain between {int(time[0][0])} and {int(time[0][-1])}.')
        
        if time[1] not in time[0]:
            raise TypeError(f'Not available data for {int(time[1])}.')
            
        data2 = data1[time[0].index(time[1]), ::-1, :]
    
    fig = plt.figure(figsize=(9, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(resolution='10m', color='black', linewidth=1)
    ax.set_extent(dataExt)
    
    if not imgExt:
        imgExt = dataExt


    if seamask:
        ocean = cpf.NaturalEarthFeature(category= 'physical', 
                                        name='ocean', scale='10m')
        
        ax.add_feature(ocean, zorder=100, edgecolor='k')

    
    if 'tempMetrics' in savePath:
        im = ax.imshow(data2[:, :], vmin=np.min(data2), vmax=np.max(data2), 
                       extent=imgExt, cmap=colorMap)

        tCols = ['Mean Bias', 'P2 Bias', 'P98 Bias', 'R (Pearson)', 
                 'std Ratio', 'RMSE', 'Bias WAMS', 'Bias CAMS']

        plt.title(f'{tCols[time[1]]}')
        plt.colorbar(im, orientation="vertical", shrink=0.5, format='%.2f', 
                     cmap=colorMap)
    
    elif 'precipMetrics' in savePath:
        maxs = [105, 205, 1, 30, 10, 100]
        mins = [-20, -50, -1, -1, -10, -50]

        if (np.min(data2) < mins[time[1]]) and (maxs[time[1]] < np.max(data2)):
            arrows = 'both'

        elif (np.min(data2) < mins[time[1]]):
            arrows = 'min'

        elif (maxs[time[1]] < np.max(data2)):
            arrows = 'max'

        else:
            arrows = None

        im = ax.imshow(data2[:, :], vmin=max(np.min(data2), mins[time[1]]), 
                       vmax=min(maxs[time[1]], np.max(data2)), extent=imgExt, 
                       cmap=colorMap)
        
        pCols = ['Mean Bias', 'P98 Bias', 'R (Spearman)', 'RMSE (Wet days)', 
                 'Bias WetAMS', 'Bias_DryAMS']

        plt.title(f'{pCols[time[1]]}')
        plt.colorbar(im, orientation="vertical", shrink=0.5, format='%.2f', 
                     cmap=colorMap, extend=arrows)
    
    elif '(mask)' in savePath:
        im = ax.imshow(data2[:, :], vmin=0, vmax=60, extent=imgExt, 
                       cmap=colorMap)

        plt.title(f'{str(time[1])[-2:]}/{str(time[1])[-4:-2]}/{str(time[1])[:4]}')
        plt.colorbar(im, orientation="vertical", shrink=0.5, format='%.2f', 
                     cmap=colorMap, extend='max')
    
    else:
        im = ax.imshow(data2[:, :], vmin=np.min(data1), vmax=np.max(data1), 
                       extent=imgExt, cmap=colorMap)
        
        plt.title(f'{str(time[1])[-2:]}/{str(time[1])[-4:-2]}/{str(time[1])[:4]}')
        plt.colorbar(im, orientation="vertical", shrink=0.5, format='%.2f', 
                     cmap=colorMap)
    
    plt.tight_layout()
    
    """
    Ensures that the file is ready to be written without the risk 
    of overwriting an existing one.
    """
    try:
        with open(savePath, 'x') as _: 
            pass  # Create an empty file to indicate write permission
    
    except PermissionError:
        raise PermissionError(f"Insufficient permissions to save the plot to '{savePath}'.")
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Save path '{savePath}' not found.")
        
    if 'anim' in savePath:
        fig.savefig(savePath, bbox_inches='tight', dpi=100)
        
    else:
        fig.savefig(savePath, bbox_inches='tight')
    
    plt.close(fig)

    return 





def animation(data, lim, var, times, frames):
    """
    Make a sequence of images of the evolution of a magnitude from maps 
    over a certain period of time.

    Parameters
    ----------
    data : np.ndarray
        Data of the magnitude being presented. 
    lim : list
        Minimum and maximum latitude and longitude with data available.
    var : str
        't' for temperature or 'p' for precipitation.
    times : list
        Dates with available data.
    frames : int
        Number of images.
        
    Raises
    ------
    TypeError
        If any parameter has an unexpected type.
    ValueError
        - If lim does not have the four coordinates needed.
        - If var is not 't' or 'p'.
        - If the number of images is greater than the number of days available.

    Returns
    -------
    None

    """
    
    # Validate input data
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a NumPy array")
      
    if not isinstance(lim, list) or len(lim) != 4:
        raise ValueError("lim must be a list with 4 coordinates (min and max latitude and longitude)")
        
    if var not in ['t', 'p']:
        raise ValueError("var must be 't' for temperature or 'p' for precipitation")
        
    if not isinstance(times, list) or not all(isinstance(i, str) for i in times):
        raise TypeError("times must be a list of strings containing the dates of the data collected")
        
    if not isinstance(frames, int) or len(times) < frames:
        raise ValueError("The number of frames must be an integer lower than the number of dates with available information")

        
    if var == 't':
        for i in tqdm(range(frames)):
            mapeo(data, [np.shape(data)[0], 68, 158] ,lim, 
                  [get_colormap(var), f'./Resultados/Temperatura/plots/anim/temperatureMap({i}).png'], 
                  [times, times[i]])
        
    else:
        for i in tqdm(range(frames)):
            mapeo(data, [np.shape(data)[0], 68, 158] ,lim, 
                  [get_colormap(var), f'./Resultados/Precipitacion/plots/anim/(mask)precipitationMap({i}).png'], 
                  [times, times[i]], True)

    return 


