import numpy as np
import tensorflow as tf
from tensorflow import keras





def temperatureModel(training, labels, lossfunc, params):
    """
    Temperature model. Contains an input layer shaped as the input's shape; 
    four hidden layers (three 2D convolutional and a flatten); an output layer 
    composed by the concatenation of two dense layers. 

    Parameters
    ----------
    training : np.ndarray
        Training data, of shape (time, latitude, longitude, geo-height)
    labels : np.ndarray
        Labels for the training data, of shape (time, outputs). 
    lossfunc : function
        Loss function to optimize.
    params : list
        List of two integers. Number of epochs and batch size.

    Returns
    -------
    model : keras.Model
        CNN model 
    history : History
        History object. Its History.history attribute is a record of training loss 
        values and metrics values at successive epochs.

    Raises
    ------
    AssertionError:
        - If arguments are not NumPy arrays.
        - If arguments are not the same length (temporal dimension).
        - If lossfunc is not a callable function.
        - If params is not a list of two integers.
        
    """

    # check input types
    assert all(isinstance(i, np.ndarray) for i in [training, labels]), "training and labels must be NumPy arrays"
    assert np.shape(training)[0] == np.shape(labels)[0], "training and labels must be the same length"
    assert callable(lossfunc), "lossfunc must be a callable function"
    assert isinstance(params, list) and len(params) == 2 and all(isinstance(i, int) for i in params), "params must be a list of two integers"
    
    
    # Input layer
    input_shape=(np.shape(training)[1], np.shape(training)[2], np.shape(training)[3])
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Hidden layers
    l1 = tf.keras.layers.Conv2D(filters=50, kernel_size=(3, 3), activation='relu', 
                                padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
    
    l2 = tf.keras.layers.Conv2D(filters=25, kernel_size=(3, 3), activation='relu', 
                                padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.01))(l1)
    
    l3 = tf.keras.layers.Conv2D(filters=10, kernel_size=(3, 3), activation='relu', 
                                padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.01))(l2)
    
    l4 = tf.keras.layers.Flatten()(l3)

    # Output layer
    l51 = tf.keras.layers.Dense(units=np.shape(labels)[1], activation='linear')(l4)
    l52 = tf.keras.layers.Dense(units=np.shape(labels)[1], activation='linear')(l4)
    outputs = tf.keras.layers.Concatenate()([l51, l52])

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss=lossfunc)
    model.summary()

    #train and store the data
    history = model.fit(training, labels, validation_split=0.1, 
                    epochs=params[0], batch_size=params[1], verbose=1)

    return model, history 





def precipModel(training, labels, lossfunc, params):
    """
    Precipitation model. Contains an input layer shaped as the input's shape; 
    four hidden layers (three 2D convolutional and a flatten); an output layer 
    composed by the concatenation of three dense layers. 

    Parameters
    ----------
    training : np.ndarray
        Data to train the model with.
    labels : np.ndarray
        Labels of the data.
    lossfunc : function
        Loss function to optimize.
    params : list
        List of two integers. Number of epochs and batch size.
        
    Returns
    -------
    model : keras.Model
        CNN model 
    history : History
        History object. Its History.history attribute is a record of training loss values and metrics values at successive epochs.
    results : np.ndarray
        Predicted values.

    Raises
    ------
    AssertionError:
        - If arguments are not NumPy arrays.
        - If arguments are not the same length (temporal dimension).
        - If lossfunc is not a callable function.
        - If params is not a list of two integers.

    """

    # check input types
    assert all(isinstance(i, np.ndarray) for i in [training, labels]), "training and labels must be NumPy arrays"
    assert np.shape(training)[0] == np.shape(labels)[0], "training and labels must be the same length"
    assert callable(lossfunc), "lossfunc must be a callable function"
    assert isinstance(params, list) and len(params) == 2 and all(isinstance(i, int) for i in params), "params must be a list of two integers"

    
    # Input layer
    input_shape=(np.shape(training)[1], np.shape(training)[2], np.shape(training)[3])
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Hidden layers
    l1 = tf.keras.layers.Conv2D(filters=50, kernel_size=(3, 3), activation='relu', 
                                padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
    
    l2 = tf.keras.layers.Conv2D(filters=25, kernel_size=(3, 3), activation='relu', 
                                padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01))(l1)
    
    l3 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), activation='relu', 
                                padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01))(l2)
    
    l4 = tf.keras.layers.Flatten()(l3)

    # Output layer
    l51 = tf.keras.layers.Dense(units=np.shape(labels)[1], activation='sigmoid')(l4)
    l52 = tf.keras.layers.Dense(units=np.shape(labels)[1], activation='linear')(l4)
    l53 = tf.keras.layers.Dense(units=np.shape(labels)[1], activation='linear')(l4)
    outputs = tf.keras.layers.Concatenate()([l51, l52, l53])

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss=lossfunc)
    model.summary()

    #train and store the data
    history = model.fit(training, labels, validation_split=0.1, 
                    epochs=params[0], batch_size=params[1], verbose=1)

    results = model.predict(training) # needed to compute the rainfall amount

    return model, history, results[:, :np.shape(labels)[1]]





# only for dense layers output !!!!!
def gaussianLoss(true, pred): 
    """
    Optimize the negative log-likelihood of the Gaussian distribution.

    Parameters
    ----------
    true : np.ndarray
        True values.
    pred : np.ndarray
        Predicted values.

    Returns
    -------
    loss : tf.python.framework.ops.EagerTensor
        Loss to be minimized.

    """

    d = int(np.shape(pred)[1]/2)
    media = pred[:, :d]
    log_var = pred[:, d:]
    precision = tf.exp(-log_var)
    loss = tf.reduce_mean(0.5 * precision * (true - media)**2 + 0.5 * log_var)
    
    return loss





def bernouilliGammaLoss(true, pred):
    """
    Optimize the negative log-likelihood of the Bernouilli-Gamma distribution.

    Parameters
    ----------
    true : np.ndarray
        True values.
    pred : np.ndarray
        Predicted values.

    Returns
    -------
    loss : tf.python.framework.ops.EagerTensor
        Loss to be minimized.

    """
    
    d = int(np.shape(pred)[1]/3)
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
    
    return loss



