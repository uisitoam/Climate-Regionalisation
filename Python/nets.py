import numpy as np
import tensorflow as tf
from keras import backend as K
from scipy.stats import gamma

# CNN for temperature
def temperatureModel(training, labels):
    """
    

    Parameters
    ----------
    training : np.ndarray
        Data to train the model with.
    labels : np.ndarray
        Labels of the data. 

    Returns
    -------
    model : keras.src.engine.functional.Functional
        CNN model 

    """
    
    # Input layer
    input_shape=(np.shape(training)[1], np.shape(training)[2], np.shape(training)[3])
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Hidden layers
    l1 = tf.keras.layers.Conv2D(filters=50, kernel_size=(3, 3), activation='relu', padding='valid')(inputs)
    l2 = tf.keras.layers.Conv2D(filters=25, kernel_size=(3, 3), activation='relu', padding='valid')(l1)
    l3 = tf.keras.layers.Conv2D(filters=10, kernel_size=(3, 3), activation='relu', padding='valid')(l2)
    l4 = tf.keras.layers.Flatten()(l3)

    # Output layer
    l51 = tf.keras.layers.Dense(units=np.shape(labels)[1], activation='linear')(l4)
    l52 = tf.keras.layers.Dense(units=np.shape(labels)[1], activation='linear')(l4)
    outputs = tf.keras.layers.Concatenate()([l51, l52])

    model = tf.keras.Model(inputs=inputs, outputs=outputs)


    return model


# CNN for precipitations
def precModel(training, labels):
    """
    

    Parameters
    ----------
    training : np.ndarray
        Data to train the model with.
    labels : np.ndarray
        Labels of the data. 

    Returns
    -------
    model : keras.src.engine.training.Model
        CNN model 

    """
    
    # Input layer
    input_shape=(np.shape(training)[1], np.shape(training)[2], np.shape(training)[3])
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Hidden layers
    l1 = tf.keras.layers.Conv2D(filters=50, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    l2 = tf.keras.layers.Conv2D(filters=25, kernel_size=(3, 3), activation='relu', padding='same')(l1)
    l3 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), activation='relu', padding='same')(l2)
    l4 = tf.keras.layers.Flatten()(l3)

    # Output layer
    l51 = tf.keras.layers.Dense(units=np.shape(labels)[1], activation='sigmoid')(l4)
    l52 = tf.keras.layers.Dense(units=np.shape(labels)[1], activation='linear')(l4)
    l53 = tf.keras.layers.Dense(units=np.shape(labels)[1], activation='linear')(l4)
    outputs = tf.keras.layers.Concatenate()([l51, l52, l53])

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model


# only for dense layers output !!!!!
def gaussianLoss(true, pred): 
    """
    

    Parameters
    ----------
    true : np.ndarray
        DESCRIPTION.
    pred : np.ndarray
        DESCRIPTION.

    Returns
    -------
    loss : tf.python.framework.ops.EagerTensor
        Loss to be minimized.

    """

    d = int(K.int_shape(pred)[1]/2)
    media = pred[:, :d]
    log_var = pred[:, d:]
    precision = tf.exp(-log_var)
    loss = tf.reduce_mean(0.5 * precision * (true - media)**2 + 0.5 * log_var)
    
    return loss

# bernoulli-gamma loss funtion (for precipitation model)
def bernouilliGammaLoss(true, pred):
    """
    

    Parameters
    ----------
    true : np.ndarray
        DESCRIPTION.
    pred : np.ndarray
        DESCRIPTION.

    Returns
    -------
    loss : tf.python.framework.ops.EagerTensor
        Loss to be minimized.

    """
    
    d = int(K.int_shape(pred)[1]/3)
    ocurrence = pred[:, :d]
    shape_parameter = tf.exp(pred[:, d:2*d])
    scale_parameter = tf.exp(pred[:,2*d:])
    bool_rain = tf.cast(tf.math.greater(true, 0), dtype=tf.float32) 
    epsilon = 0.000001 #avoid indeterminations
    loss = -tf.reduce_mean((1-bool_rain)*tf.math.log(1 - ocurrence + epsilon) + 
                           bool_rain*(tf.math.log(ocurrence+epsilon) 
                                      + (shape_parameter - 1)*tf.math.log(true + epsilon) 
                                      - shape_parameter*tf.math.log(scale_parameter + epsilon)
                                      - tf.math.lgamma(shape_parameter + epsilon) 
                                      - true/(scale_parameter + epsilon))) #logarithm of the b-g loss
    
    print(f'Type of precip loss: {loss}')
    
    return loss


#simulate = True es stochastic
def computeRainfall(log_alpha, log_beta, simulate = False, bias = None):
    
    if simulate:
        aux = np.zeros((np.shape(log_alpha)[0], np.shape(log_alpha)[1]))
        alpha = tf.exp(log_alpha)
        beta = tf.exp(log_beta)
        
        for i in range(np.shape(log_alpha)[1]):  
            aux[:, i] = gamma.rvs(alpha[:, i], beta[:, i], size=np.shape(log_alpha)[0])
        
        amoData = aux
    
    else:
        amoData = tf.exp(log_alpha)*tf.exp(log_beta) 
        
    if bias is not None:
        amoData += 1
        
    return amoData
    



