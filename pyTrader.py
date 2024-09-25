
"""
@author__ = "Juan Francisco Illan"
@license__ = "GPL"
@version__ = "1.0.1"
@email__ = "juanfrancisco.illan@gmail.com"
"""

import pandas as pd
import numpy as np
import time

import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, SimpleRNN, Dense, Flatten, Embedding, LSTM, Bidirectional, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import SGD, Adam, Adadelta, RMSprop
from keras.models import load_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler

def createAdvisorMultivariateLSTM(datasetTrain,datasetTest, time_step, variableInput, variableOutput, escalado):  
    
    initial_train_time = time.time()
    
    set_x_entrenamiento = datasetTrain[variableInput]  # CLOSE COLUMN
    set_y_entrenamiento = datasetTrain[variableOutput] # CLOSE COLUMN

    set_x_validacion = datasetTest[variableInput] # CLOSE COLUMN1
    set_y_validacion = datasetTest[variableOutput] # CLOSE COLUMN1

    sc = MinMaxScaler(feature_range=(0,1))
    sc_y = MinMaxScaler(feature_range=(0,1))
        
    if (escalado):
        # Es necesario escalar los datos, para que relativamente todos tengan 
        # significado relativo entre distintos valores
        # Escalamos entre -1 y 1
        set_x_entrenamiento_escalado = sc.fit_transform(np.array(set_x_entrenamiento).reshape(-1,1))        
        set_x_validacion_escalado = sc.transform(np.array(set_x_validacion).reshape(-1,1))
        set_y_entrenamiento_escalado = sc_y.fit_transform(np.array(set_y_entrenamiento).reshape(-1,1))
        set_y_validacion_escalado = sc_y.transform(np.array(set_y_validacion).reshape(-1,1))
    else :
        set_x_entrenamiento_escalado = set_x_entrenamiento
        set_y_entrenamiento_escalado = set_y_entrenamiento
        set_x_validacion_escalado = set_x_validacion
        set_y_validacion_escalado = set_y_validacion
            
    

    X_train = []
    Y_train = []
    m = len(set_x_entrenamiento_escalado)

    #usamos el bloque for para iterativamente dividir el set de entrenamiento en bloques de {time_step} datos
    for i in range(time_step,m-1):
        # X: bloques de tamaño "time_step" datos: 
        # desde 0 : time_step --> time_step+1 , es decir [0:10] --> [11]
        # desde 1 : time_step+1 --> time_step+2
        # desde 2 : time_step+2 --> time_step+3
        X_train.append(set_x_entrenamiento_escalado[i-time_step:i])
        # Y: el el dato de BUY o SELL de proxima sesion (i+1)
        Y_train.append(set_y_entrenamiento_escalado[i:i+1])

    X_train, Y_train = np.array(X_train), np.array(Y_train)

    # Antes de crear la Red LSTM debemos reajustar los sets que acabamos de obtener, 
    # para indicar que cada ejemplo de entrenamiento 
    # a la entrada del modelo será un vector de 10x1. Para esto usamos la función reshape de Numpy:
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    dim_entrada = (X_train.shape[1],1) # Tamaño de entrada LSTM = time_step x 1
    dim_salida = 1 # Numero de pasos que a futuro debe predecir el modelo LSTM
    n_unit = 128 # Tamaño del estado oculto del modelo
    
    # Cargar el modelo desde disco
    #model = load_model('modelo_keras_lstm.h5')
    
    model = Sequential()
    model.add(LSTM(units=n_unit, input_shape=dim_entrada))
    #model.add(LSTM(units=n_unit, return_sequences = True, input_shape=dim_entrada))
    #model.add(LSTM(50,return_sequences = True,input_shape = (X_train.shape[1],1)))
    #model.add(LSTM(units=n_unit,return_sequences = True))
    #model.add(LSTM(units=n_unit))
    #model.add(Dense(1)) 
    model.add(Dense(units=dim_salida, activation='linear')) 
    # activacion linear para predecir datos de forma lineal entre -1 y 1
    #model.compile(optimizer='rmsprop', loss='mse')
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')
    
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    print(short_model_summary)
    
    model.fit(X_train,Y_train,epochs=20,batch_size=256, 
               verbose=2)
    
    # Guarda el modelo en disco
    model.save('modelo_keras_lstm.h5')

    final_train_time = time.time()

    # El batch size en Deep Learning forma parte de la optimización de hiperparámetros que se aplican en el estudio de las redes neuronales profundas para el manejo de los macrodatos.
    # De hecho, de entre todos los hiperparámetros, el learning rate y el batch size son dos parámetros directamente relacionados con el algoritmo del gradient descent.

    X_test = []
    Y_test = []
    for i in range(time_step,len(set_x_validacion_escalado)-1):
        X_test.append(set_x_validacion_escalado[i-time_step:i])  
        Y_test.append(set_y_validacion_escalado[i:i+1])
        # [0:10] --> [11]
        # [1:11] --> [12]
        # [2:12] --> [13]
    X_test, Y_test = np.array(X_test), np.array(Y_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

    # Evaluar el modelo
    #score = model.evaluate(X_test, Y_test)
    #print('Test loss:', score[0])
    #print('Test accuracy:', score[1])
 
    prediccion = model.predict(X_test)
    #print(confusion_matrix(prediccion, Y_test))
    if (escalado) :
        prediccion = sc_y.inverse_transform(prediccion)

    # Time execution
    final_test_time = time.time()
    time_execution = final_test_time - initial_train_time

    return model, prediccion


def createAdvisorLSTM(datasetTrain,datasetTest, time_step, variableInput, variableOutput):  
    
    initial_train_time = time.time()
    
    set_x_entrenamiento = datasetTrain[variableInput]  # CLOSE COLUMN
    set_y_entrenamiento = datasetTrain[variableOutput] # CLOSE COLUMN

    set_x_validacion = datasetTest[variableInput] # CLOSE COLUMN1
    set_y_validacion = datasetTest[variableOutput] # CLOSE COLUMN1

    # Es necesario escalar los datos, para que relativamente todos tengan 
    # significado relativo entre distintos valores
    # Escalamos entre -1 y 1
    sc = MinMaxScaler(feature_range=(-1,1))
    set_x_entrenamiento_escalado = sc.fit_transform(np.array(set_x_entrenamiento).reshape(-1,1))
    set_y_entrenamiento_escalado = sc.fit_transform(np.array(set_y_entrenamiento).reshape(-1,1))
    set_x_validacion_escalado = sc.fit_transform(np.array(set_x_validacion).reshape(-1,1))
    set_y_validacion_escalado = sc.fit_transform(np.array(set_y_validacion).reshape(-1,1))

    X_train = []
    Y_train = []
    m = len(set_x_entrenamiento_escalado)

    #usamos el bloque for para iterativamente dividir el set de entrenamiento en bloques de {time_step} datos
    for i in range(time_step,m-1):
        # X: bloques de tamaño "time_step" datos: 
        # desde 0 : time_step --> time_step+1 , es decir [0:10] --> [11]
        # desde 1 : time_step+1 --> time_step+2
        # desde 2 : time_step+2 --> time_step+3
        X_train.append(set_x_entrenamiento_escalado[i-time_step:i])
        # Y: el el dato de BUY o SELL de proxima sesion (i+1)
        Y_train.append(set_y_entrenamiento_escalado[i:i+1])

    X_train, Y_train = np.array(X_train), np.array(Y_train)

    # Antes de crear la Red LSTM debemos reajustar los sets que acabamos de obtener, 
    # para indicar que cada ejemplo de entrenamiento 
    # a la entrada del modelo será un vector de 10x1. Para esto usamos la función reshape de Numpy:
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


    dim_entrada = (X_train.shape[1],1) # Tamaño de entrada LSTM = time_step x 1
    dim_salida = 1 # Numero de pasos que a futuro debe predecir el modelo LSTM
    n_unit = 50 # Tamaño del estado oculto del modelo
    
    # Cargar el modelo desde disco
    #model = load_model('modelo_keras_lstm.h5')
    
    model = Sequential()
    model.add(LSTM(units=n_unit, return_sequences = True, input_shape=dim_entrada))
    #model.add(LSTM(50,return_sequences = True,input_shape = (X_train.shape[1],1)))
    #model.add(LSTM(units=n_unit,return_sequences = True))
    #model.add(LSTM(units=n_unit))
    #model.add(Dense(1)) 
    model.add(Dense(units=dim_salida, activation='linear')) 
    # activacion linear para predecir datos de forma lineal entre -1 y 1
    #model.compile(optimizer='rmsprop', loss='mse')
    model.compile(loss = 'mean_squared_error',optimizer = 'adam')
    
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    print(short_model_summary)
    
    model.fit(X_train,Y_train,epochs=20,batch_size=256, 
               verbose=2)
    
    # Guarda el modelo en disco
    model.save('modelo_keras_lstm.h5')

    final_train_time = time.time()

    # El batch size en Deep Learning forma parte de la optimización de hiperparámetros que se aplican en el estudio de las redes neuronales profundas para el manejo de los macrodatos.
    # De hecho, de entre todos los hiperparámetros, el learning rate y el batch size son dos parámetros directamente relacionados con el algoritmo del gradient descent.

    X_test = []
    Y_test = []
    for i in range(time_step,len(set_x_validacion_escalado)-1):
        X_test.append(set_x_validacion_escalado[i-time_step:i])  
        Y_test.append(set_y_validacion_escalado[i:i+1])
        # [0:10] --> [11]
        # [1:11] --> [12]
        # [2:12] --> [13]
    X_test, Y_test = np.array(X_test), np.array(Y_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

    # Evaluar el modelo
    score = model.evaluate(X_test, Y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
 
    prediccion = model.predict(X_test)
    print(confusion_matrix(prediccion, Y_test))
    prediccion = sc.inverse_transform(prediccion)

    # Time execution
    final_test_time = time.time()
    time_execution = final_test_time - initial_train_time

    return model, prediccion
