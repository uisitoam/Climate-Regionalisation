import numpy as np 
import matplotlib.pyplot as plt
from nets import temperatureModel, precipModel

colores = ['#FF0000', '#FF9B00', '#F5DE00', '#51FF53', '#00A902', '#00FF8F', 
           '#00F5E6', '#00B2FF', '#0032FF', '#B900FF', '#8700FF', '#FF86F6', 
           '#EA00D9', '#B00070', '#A4A4A4']

def learningCurve(model, ejey, ep):
    fig, ax = plt.subplots(1, 1, figsize=(9,5))

    ejex = np.arange(1, ep + 1)

    ax.plot(ejex, ejey[0], color=colores[5], label='Training Loss')
    ax.plot(ejex, ejey[1], color=colores[9], label='Validation Loss')
    ax.set(xlabel='Epochs', ylabel='Loss', title='Training and Validation Loss')
    ax.legend()
    
    if model == temperatureModel:
        fig.savefig(f'./Resultados/Temperatura/plots/{ep}_epochsLearning_curve(temp).pdf')
        
    elif model == precipModel:
        fig.savefig(f'./Resultados/Precipitacion/plots/{ep}_epochsLearning_curve(precip).pdf')
    
    return 





def boxplots(metrics, metricLabels, params):
    def approx(value, expect):
      resto = value % expect
      
      if value < 0:
        return value - resto
    
      else:
        return value + (expect - resto)
    
    data = [element for element in metrics]
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 10))
    
    ax.boxplot(data, labels=metricLabels)
    
    ax.set(title=params[0])
    ax.set_xticklabels(metricLabels, rotation=45)
    
    if params[0] == 'Temperature':
        step = 0.5
        yticks = np.arange(approx(np.min(data), step) - step, 
                           approx(np.max(data), step) + step, step)
        
        ylabels = [f'{y:1.1f}' for y in yticks]
        ax.set_yticks(yticks, minor=True, labels=ylabels)
        ax.tick_params(axis='y', labelsize=10)
        
    elif params[0] == 'Precipitation':
        step = 5
        yticks = np.arange(approx(np.min(data), step) - step, 
                           approx(np.max(data), step) + step, step)
        
        ylabels = [f'{y:1.1f}' for y in yticks]
        ax.set_yticks(yticks, labels=ylabels, minor=True)
        ax.tick_params(axis='y', which='both', labelsize=8)
    
    else: 
        raise TypeError("Title must be `Temperature` or `Precipitation`")
    
    plt.subplots_adjust(top=0.935, bottom=0.175, left=0.12, right=0.88, 
                        hspace=0.2, wspace=0.2)
    
    fig.savefig(params[1])
    
    return 




