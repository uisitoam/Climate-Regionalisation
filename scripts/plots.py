import os
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import cartopy.feature as cpf
import cartopy.crs as ccrs
from tqdm import tqdm
import reader as rd





def save_plot(figura, ruta, nombre):
    """
    Guardar la figura en la ruta especificada.

    Parameters
    ----------
    ruta : str
        Ruta donde se guardará la figura.
    nombre : str
        Nombre del archivo.
    
    Returns
    -------
    None

    Raises
    ------
    AssertionError
        Si la ruta o el nombre no son strings.

    """

    # check input parameters
    assert isinstance(ruta, str), "ruta must be a a string."
    assert isinstance(nombre, str), "nombre must be a string."


    # check if the folder exists
    try:
        if not os.path.exists(ruta):
            os.makedirs(ruta) # if it does not exist, create it

    except Exception as e:
        raise Exception(f"Error al crear la carpeta: {e}")

    # check if the file exists
    filename = os.path.join(ruta, nombre)
    if os.path.exists(filename):
        base, ext = os.path.splitext(nombre) # if it does, add a version number

        i = 1
        while os.path.exists(os.path.join(ruta, f"{base}_v{i}{ext}")):
            i += 1

        filename = os.path.join(ruta, f"{base}_v{i}{ext}")

    # save the figure
    if 'anim' in ruta:
        figura.savefig(filename, dpi=100)
    
    else:
        figura.savefig(filename)

    plt.close(figura)

    return 





def learningCurve(ejey, savePath, guardar=save_plot):
    """
    This functions plots the learning curves of the models. 

    Parameters
    ----------
    ejey : list
        List containing the training and the validation loss as two different 
        NumPy arrays. 
    savePath : list
        List containing the path and the name of the file where the plot will be saved.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        - If ejey is not a list of two NumPy arrays.
        - If savePath is not a list of two strings.

    """

    # check input parameters
    assert len(ejey) == 2 and all(isinstance(i, np.ndarray) for i in ejey), "ejey must be a list of 2 NumPy arrays."
    assert len(savePath) == 2 and all(isinstance(i, str) for i in savePath), "savePath must be a list of two strings: path and name of the file."


    fig, ax = plt.subplots(1, 1, figsize=(9,5))

    ejex = np.arange(1, len(ejey[0]) + 1)

    ax.plot(ejex, ejey[0], color='#00FF8F', label='Training Loss')
    ax.plot(ejex, ejey[1], color='#B900FF', label='Validation Loss')
    ax.set(xlabel='Epochs', ylabel='Loss', title='Training and Validation Loss')
    ax.legend()

    guardar(fig, *savePath)
    
    return 





def boxplots(metrics, metricLabels, params, guardar=save_plot):
    """
    This function plots boxplots of the metrics.

    Parameters
    ----------
    metrics : np.ndarray
        2D array containing the metrics to be plotted.
    metricLabels : list
        List containing the names of the metrics.
    params : list
        List containing the parameters for the plot: 
        - The type of plot (temp, pr, tempMets, prMets, alltogether, alltogether_metrics).
        - The width of the plot.
        - The height of the plot.
        - The title of the plot.
        - The path where the plot will be saved.
        - The name of the file.

    Returns
    -------
    None

    Raises
    ------
    None
        
    """   

    # approximate the values to the nearest multiple (for the xticks)
    def approx(value, expect):
      resto = value % expect
      
      if value < 0:
        return value - resto
    
      else:
        return value + (expect - resto)
      
    
    data = np.transpose(metrics) # boxplot paints a box for each column of the array
    # alternativa: data = [elem for elem in metrics] # boxplot paints a box for each element of the list

    fig, ax = plt.subplots(1, 1, figsize=(params[1], params[2])) # 9, 5

    orientacion = False if params[0] in ['temp', 'pr'] else True # horizontal if temp or pr, vertical otherwise

    ax.boxplot(data, vert=orientacion, showfliers=False, notch=True, labels=metricLabels) if 'alltogether' not in params[0] else None

    # set the labels and ticks for temperature and precipitation plots
    if params[0] == 'temp' or params[0] == 'pr':
        ax.set_yticklabels(metricLabels)
        
        q1 = np.nanquantile(metrics, 0.25, axis=1)
        q3 = np.nanquantile(metrics, 0.75, axis=1)
        iqr = q3 - q1

        step = 1 if params[0] == 'temp' else 50
        xticks = np.arange(approx(np.min(q1 - 1.5*iqr), step), 
                           approx(np.max(q3 + 1.5*iqr), step), step)
        
        xlabels = [f'{x:1.1f}' for x in xticks]
        ax.set_xticks(xticks, minor=True, labels=xlabels)
        ax.tick_params(axis='x', labelsize=14)
        fig.subplots_adjust(top=0.92, bottom=0.1, left=0.15, right=0.95)
    
    # all in one plots
    elif 'alltogether' in params[0]:
        color_map = {'temp': {20: ["#81FF72"] * 6 + ['#D8CC3C'] * 2 + ["#FFBB72"] * 6 + ["#FF7272"] * 6,
                              18: ["#81FF72"] * 6 + ["#FFBB72"] * 6 + ["#FF7272"] * 6,
                              12: ["#81FF72"] * 4 + ["#FFBB72"] * 4 + ["#FF7272"] * 4,
                              10: ["#81FF72"] * 3 + ['#D8CC3C'] + ["#FFBB72"] * 3 + ["#FF7272"] * 3,
                              9: ["#81FF72"] * 3 + ["#FFBB72"] * 3 + ["#FF7272"] * 3, 
                              3: ["#81FF72", "#FFBB72", "#FF7272"]},
                     'pr': {20: ["#3BE7D5"] * 6 + ['#2FB4A6'] * 2 + ["#D8CC3C"] * 6 + ["#B78928"] * 6,
                            18: ["#3BE7D5"] * 6 + ["#D8CC3C"] * 6 + ["#B78928"] * 6,
                            12: ["#3BE7D5"] * 4 + ["#D8CC3C"] * 4 + ["#B78928"] * 4,
                            10: ["#3BE7D5"] * 3 + ['#2FB4A6'] + ["#D8CC3C"] * 3 + ["#B78928"] * 3,
                            9: ["#3BE7D5"] * 3 + ["#D8CC3C"] * 3 + ["#B78928"] * 3, 
                            3: ["#81FF72", "#FFBB72", "#FF7272"]}}

        param = 'temp' if 'temp' in params[0] else 'pr'

        if param in color_map and len(metricLabels) in color_map[param]:
            colors = color_map[param][len(metricLabels)]

        if 'metrics' in params[0]: # metrics plots
            for i in range(len(metricLabels)):
                bp = ax.boxplot(data[:, i:i+1], positions=[i], vert=orientacion, showfliers=False, notch=True, 
                                patch_artist=True, boxprops=dict(facecolor=colors[i]), 
                                medianprops=dict(color="#000000"), widths=(0.5))
            
            ax.set_xticks(range(len(metricLabels)))
            loc = 'best'
        
        else: # all results in one plot
            for i in range(len(metricLabels)):
                bp = ax.boxplot(data[:, i:i+1], positions=[i], vert=orientacion, showfliers=False, notch=True, 
                                patch_artist=True, boxprops=dict(facecolor=colors[i]), 
                                medianprops=dict(color="#000000"), widths=(0.5))
            
            ax.set_xticks(range(len(metricLabels)))

            if 'temp' in params[0]:
                ax.set_ylabel('Mean temperature [ºC]', fontsize=16)  # Cambia 'large' a tu tamaño de fuente preferido
                loc = 'lower right'
            
            else:
                ax.set_ylabel('Precipitation amount[mm/day]', fontsize=16)
                loc = 'upper right'

        ax.tick_params(axis='y', labelsize=15)

        ax.set_xticklabels(metricLabels, rotation=45, fontsize=15)


        colores = list(set(colors))

        if 'temp' in params[0]:
            coloritos = ["#81FF72", '#D8CC3C', "#FFBB72", "#FF7272"]
        
        elif 'pr' in params[0]:
            coloritos = ["#3BE7D5", '#2FB4A6', "#D8CC3C", "#B78928"]

        if len(colores) == 4:
            patch1 = mpatches.Patch(color=coloritos[0], label='1980-2009')
            patch2 = mpatches.Patch(color=coloritos[1], label='2010-2019')
            patch3 = mpatches.Patch(color=coloritos[2], label='2030-2059')
            patch4 = mpatches.Patch(color=coloritos[3], label='2070-2099')

            ax.legend(handles=[patch1, patch2, patch3, patch4], loc=loc, fontsize=12)
        
        else: # metrics plots
            patch1 = mpatches.Patch(color=coloritos[0], label='1980-2009')
            patch2 = mpatches.Patch(color=coloritos[2], label='2030-2059')
            patch3 = mpatches.Patch(color=coloritos[3], label='2070-2099')

            ax.legend(handles=[patch1, patch2, patch3], loc=loc, fontsize=12)

        fig.subplots_adjust(top=0.92, bottom=0.2, left=0.05, right=0.95)

    else:
        ax.set_xticks(range(1, len(metricLabels) + 1) )
        ax.set_xticklabels(metricLabels, fontsize=15)
    
    ax.set(title=params[3])

    guardar(fig, params[4], params[5])
    
    return 





def get_colormap(var):
    """
    This function returns the colormap for the plots.

    Parameters
    ----------
    var : str
        The variable to be plotted: 'temp' for temperature, 'pr' for precipitation, 'tempMets' for temperature metrics, 
        'prMets' for precipitation metrics.

    Returns
    -------
    LinearSegmentedColormap or str
        The colormap for the plot.

    Raises
    ------
    AssertionError
        If var is not 'temp', 'pr', 'tempMets', or 'prMets'.

    """

    # check input parameters
    assert var in ('temp', 'pr', 'tempMets', 'prMets'), "Variable must be 'temp' or 'tempMets' for temperature or 'pr' or 'prMets' for precipitation."


    colormaps = {'temp': 'jet',
                 'pr': [(1.000, 1.000, 1.000), (0.929, 0.929, 0.964), (0.857, 0.857, 0.929), 
                       (0.786, 0.786, 0.893), (0.714, 0.714, 0.857), (0.643, 0.643, 0.821),
                       (0.571, 0.571, 0.786), (0.500, 0.500, 0.750), (0.429, 0.429, 0.714),
                       (0.357, 0.357, 0.679), (0.286, 0.286, 0.643), (0.214, 0.214, 0.607),
                       (0.143, 0.143, 0.571), (0.071, 0.071, 0.536), (0.000, 0.000, 0.500),
                       (0.000, 0.000, 0.500)],
                 'tempMets': 'OrRd', 'prMets': 'BrBG'}
    
    if var == 'pr':
        return LinearSegmentedColormap.from_list("full_spectrum", colormaps[var])
    
    else: 
        return colormaps[var]




    
def mapeo(data, dates, time, params, cotas=None, 
             dataExt=[np.min(rd.maskLon), np.max(rd.maskLon), np.min(rd.maskLat), np.max(rd.maskLat)], 
             seamask=True, get=get_colormap, guardar=save_plot):
    """
    This function plots different maps of the data.

    Parameters
    ----------
    data : np.ndarray
        3D array containing the data to be plotted.
    dates : np.ndarray
        1D array containing the dates of the data.
    time : int
        The index of the date to be plotted.
    params : list
        List containing the parameters for the plot: 
        - The type of plot (temp, pr, tempMets, prMets).
        - The title of the plot.
        - The path where the plot will be saved.
        - The name of the file.
    cotas : list, optional
        List containing the minimum and maximum values for the plot. The default is None.
    dataExt : list, optional
        List containing the minimum and maximum values for the x and y axes. 
        The default is [np.min(rd.maskLon), np.max(rd.maskLon), np.min(rd.maskLat), np.max(rd.maskLat)].
    seamask : bool, optional
        A boolean flag to enable the sea mask. The default is True.
    get : function, optional
        A function to get the colormap. The default is get_colormap.
    guardar : function, optional
        A function to save the plot. The default is save_plot.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        - If data is not a 3D NumPy array.
        - If dataExt is not a list of four floats.
        - If params is not a list of four elements.
        - If dates is not a NumPy array.

    """

    # check input parameters
    assert isinstance(data, np.ndarray) and data.ndim == 3, "Data must be a 3D NumPy array."
    assert isinstance(dataExt, list) and len(dataExt) == 4 and all(isinstance(i, np.float32) for i in dataExt), "dataExt must be a list of 4 floats."
    assert isinstance(params, list) and len(params) == 4, "Params must be a tuple of length 4 containing (colormap name, plot title, save path, file name)."
    assert isinstance(dates, np.ndarray), "dates must be a NumPy array."
  

    colorMapName, title, savePath, fileName = params
    colorMap = get(colorMapName)

    data = data - 273.15 if colorMapName == 'temp' else data
    data2 = data[list(dates).index(time), ::-1, :] if 'Mets' not in colorMapName else data[time, ::-1, :]

    #min, max = cotas if cotas is not None else [np.min(data), np.max(data)]
    if colorMapName == 'temp'  or colorMapName == 'pr':
        min_val, max_val = [np.min(data[data > -100]), np.max(data)]
        vmin = 0 if colorMapName != 'pr' else 0
        vmax = 30 if colorMapName != 'pr' else 40

    elif 'Mets' in colorMapName:
        min_val, max_val = cotas
        vmin = max(np.min(data2), min_val)
        vmax = min(np.max(data2), max_val)
    
    else:
        min_val, max_val = [np.min(data2), np.max(data2)]
        vmin = max(np.min(data2), min_val)
        vmax = min(np.max(data2), max_val)
    

    fig = plt.figure(figsize=(9, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(resolution='10m', color='black', linewidth=1)
    ax.set_extent(dataExt)

    if seamask:
        ocean = cpf.NaturalEarthFeature(category = 'physical', 
                                        name = 'ocean', scale = '10m')
        
        ax.add_feature(ocean, zorder=100, edgecolor='k', facecolor='#254D77')
    
    im = ax.imshow(data2[:, :], vmin=vmin, vmax=vmax, extent=dataExt, cmap=colorMap)
    
    if (np.min(data2[data2 > -100]) < min_val) and (max_val < np.max(data2)):
            arrows = 'both'

    elif (np.min(data2[data2 > -100]) < min_val):
        arrows = 'min'

    elif (max_val < np.max(data2)):
        arrows = 'max'

    else:
        arrows = None

    if colorMapName == 'temp':
        arrows = 'both'  
    
    elif colorMapName == 'pr':
        arrows = 'max'

    plt.title(f'{title}')

    cbar = plt.colorbar(im, orientation="vertical", shrink=0.5, format='%.2f', 
                     cmap=colorMap, extend=arrows)
    
    cbar.ax.tick_params(labelsize=12)
    
    if colorMapName == 'temp':
        cbar_label = 'Mean Temperature [ºC]'
    
    elif colorMapName == 'pr':
        cbar_label = 'Precipitation amount [mm/day]'
    
    else:
        cbar_label = 'Metric value'
    
    cbar.set_label(cbar_label, rotation=270, labelpad=25, fontsize=12)
    
    plt.tight_layout()

    guardar(fig, savePath, fileName)

    return 





def animation(data, dates, params, frames=365):
    """
    This function generates an multiple images of the data to animate it.

    Parameters
    ----------
    data : np.ndarray
        3D array containing the data to be plotted.
    dates : np.ndarray
        1D array containing the dates of the data.
    params : list
        List containing the parameters for the plot: 
        - The type of plot (temp, pr).
        - The path where the plot will be saved.
        - The name of the file.
    frames : int, optional
        The number of frames for the animation. The default is 365.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        - If data is not a 3D NumPy array.
        - If params is not a list of three elements.
        - If dates is not a NumPy array.
        - If frames is not an integer.

    """

    # check input parameters
    assert isinstance(data, np.ndarray), "data must be a NumPy array"
    assert isinstance(params, list) and len(params) == 3, "params must be a list of three elements"
    assert all(isinstance(p, str) for p in params), "params must be a list of three strings"
    assert params[0] in ['temp', 'pr'], "params[0] must be 'temp' for temperature or 'pr' for precipitation"
    assert isinstance(frames, int), "frames must be an integer"

    
    var, savePath, fileName = params

    title = r'Temperature on ' if var == 'temp' else r'Precipitation amount on '

    if params[0] == 'temp': # temperature
        for i in tqdm(range(frames)):
            mapeo(data, dates, dates[-(frames - i)], [var, title + dates[i], savePath, fileName + f'{i}.png'])

    else: # precipitation
        for i in tqdm(range(frames)):
            mapeo(data, dates, dates[-(frames - i)], [var, title + dates[i], savePath, fileName + f'{i}.png'])

    return 





def comparacion(data, params, plot=boxplots):
    """
    This function plots the boxplots of the metrics for the different models.

    Parameters
    ----------
    data : list
        List containing the metrics for the different models.
    params : list
        List containing the parameters for the plot: 
        - The title of the plot.
        - The path where the plot will be saved.
        - The name of the file.
    plot : function, optional
        The function to plot the boxplots. The default is boxplots.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        - If data is not a list of NumPy arrays.
        - If data is not a list of 3 elements.
        - If the first element of data is not a 2D NumPy array.
        - If the second element of data is not a list of strings.
        - If params is not a list of three strings.
        - If plot is not a function.

    """

    # check input parameters
    assert isinstance(data, list) and all(isinstance(i, np.ndarray) for i in data), "data must be a list of NumPy arrays."
    assert isinstance(params, list) and len(params) == 3, "params must be a list of three elements"
    assert all(isinstance(p, str) for p in params), "params must be a list of three strings"
    assert callable(plot), "plot must be a function"


    if 'Temperatura' in params[1]:
        metricas = ['Mean Bias', 'P2 Bias', 'P98 Bias', 'R Pearson', 'std ratio', 'RMSE', 'WAMS', 'CAMS']
        variable = 'temp'
    
    elif 'Precipitacion' in params[1]:
        metricas = ['Mean Bias', 'P98 Bias', 'R Spearman', 'RMSE (Wet Days)', 'Wet AMS', 'Dry AMS']
        variable = 'pr'

    for i in range(np.shape(data[0])[0]):
        datos = [np.reshape(data[j][i, :], (1, -1)) for j in range(len(data))]
        total = np.concatenate(datos, axis=0)
        title = params[0] + f' {metricas[i]}'
        fileName = title
        plot(total, ['GFDL', 'IPSL', 'MIROC', 'Era5', 'GFDL', 'IPSL', 'MIROC', 'GFDL', 'IPSL', 'MIROC'], 
             [f'alltogether_metrics_{variable}', 9, 6, title, params[1], f'{fileName}.pdf'])

    return 





def big_boxplot(data, var, plot=boxplots):
    """
    This function plots the boxplots of the predictions for the different models.
    
    Parameters
    ----------
    data : list
        List containing the predictions for the different models.
    var : str
        The variable to be plotted: 'temp' for temperature, 'pr' for precipitation.
    plot : function, optional
        The function to plot the boxplots. The default is boxplots.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        - If data is not a list of NumPy arrays.
        - If var is not 'temp' or 'pr'.
        - If plot is not a function.
        
    """

    # check input parameters
    assert isinstance(data, list) and all(isinstance(i, np.ndarray) for i in data), "data must be a list of NumPy arrays."
    assert var in ['temp', 'pr'], "var must be 'temp' for temperature or 'pr' for precipitation."
    assert callable(plot), "plot must be a function."


    # fix some parameters for each variable
    if var == 'temp':
        title = 'Temperature evolution'
        savePath = 'Resultados/Temperatura/plots/'
        correccion = 273.15
        tipo = 'alltogether_temp'
    
    elif var == 'pr':
        title = 'Precipitation evolution'
        savePath = 'Resultados/Precipitacion/plots/'
        correccion = 0
        tipo = 'alltogether_pr'

    datos = [0] * len(data)

    for i in range(len(data)):
        datos[i] = np.reshape(np.mean(data[i], axis=0), (1, -1))
    
    total = np.concatenate(datos, axis=0)

    plot(total - correccion, ['GFDL-WRF', 'GFDL-CNN', 'IPSL-WRF', 'IPSL-CNN', 'MIROC-WRF', 'MIROC-CNN', 'Era5-WRF', 'Era5-CNN',
                 'GFDL-WRF', 'GFDL-CNN', 'IPSL-WRF', 'IPSL-CNN', 'MIROC-WRF', 'MIROC-CNN', 
                 'GFDL-WRF', 'GFDL-CNN', 'IPSL-WRF', 'IPSL-CNN', 'MIROC-WRF', 'MIROC-CNN'], 
                 [tipo, 14, 7, title, savePath, 'predicciones_comparativa.pdf'])
    
    return