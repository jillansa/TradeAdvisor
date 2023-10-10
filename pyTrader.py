
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

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler

from neuronal_helper import *

def createAdvisorLSTM(dataset,datasetTest,dataPredict):
    
    
    set_entrenamiento = dataset.iloc[:,1:2]
    set_validacion = datasetTest.iloc[:,1:2]
    set_predict = dataPredict.iloc[:,1:2]


    sc = MinMaxScaler(feature_range=(0,1))
    set_entrenamiento_escalado = sc.fit_transform(set_entrenamiento)


    time_step = 60
    X_train = []
    Y_train = []
    m = len(set_entrenamiento_escalado)

    #usamos el bloque for para iterativamente dividir el set de entrenamiento en bloques de 60 datos
    for i in range(time_step,m):
        # X: bloques de "time_step" datos: 0-time_step, 1-time_step+1, 2-time_step+2, etc
        X_train.append(set_entrenamiento_escalado[i-time_step:i,0])
        # Y: el siguiente dato
        Y_train.append(set_entrenamiento_escalado[i,0])

    X_train, Y_train = np.array(X_train), np.array(Y_train)

    # Antes de crear la Red LSTM debemos reajustar los sets que acabamos de obtener, para indicar que cada ejemplo de entrenamiento 
    # a la entrada del modelo será un vector de 60x1. Para esto usamos la función reshape de Numpy:
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


    dim_entrada = (X_train.shape[1],1)
    dim_salida = 1
    na = 50
    modelo = Sequential()
    modelo.add(LSTM(units=na, input_shape=dim_entrada))
    modelo.add(Dense(units=dim_salida))
    modelo.compile(optimizer='rmsprop', loss='mse')
    modelo.fit(X_train,Y_train,epochs=20,batch_size=32)


    x_test = set_validacion.values
    x_test = sc.transform(x_test)

    X_test = []
    for i in range(time_step,len(x_test)):
        X_test.append(x_test[i-time_step:i,0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

    prediccion = modelo.predict(X_test)
    prediccion = sc.inverse_transform(prediccion)

    return prediccion
    

def createClasiffierLSTM(seq_data):

    time_fit = 0.0
    time_test = 0.0

    start_fit_time = time.time()
    seq_data['words'] = seq_data.apply(lambda x: getKmers(x['SEQUENCE'],6), axis=1)
    seq_data = seq_data.drop('SEQUENCE', axis=1)

    #seq_data.head()
    seq_texts = list(seq_data['words'])
    for item in range(len(seq_texts)):
        seq_texts[item] = ' '.join(seq_texts[item])
    labels = seq_data.iloc[:, 2].values # PROTEIN_ID

    tokenizer = Tokenizer() 
    tokenizer.fit_on_texts(seq_texts) # tokenizer the word in each secuence
    encoded_docs = tokenizer.texts_to_sequences(seq_texts) # Transform unique each token in a integer value
    max_length = max([len(s) for s in encoded_docs]) # 135 max langth of all secuences
    X = pad_sequences(encoded_docs, maxlen = max_length, padding = 'post') # the context is determinate in less 100 nucleotid

    X_train,X_test,y_train,y_test = train_test_split(X,labels,
                                                    test_size=0.20,random_state=42)
    vocab_size = len(tokenizer.word_index) + 1 # Word padding
    print(X_train.shape)
    print(X_test.shape)

    # Keras ofrece una capa de incrustación que se puede utilizar para redes neuronales en datos de texto. 
    # Requiere que los datos de entrada estén codificados en números enteros, 
    # de modo que cada palabra esté representada por un número entero único.
    #n_timesteps, n_features, n_outputs = X_train.shape[0], X_train.shape[1], y_train[0].shape

    model = Sequential()
    model.add(Embedding(vocab_size, 32))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(1, activation='sigmoid'))
    

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    #history_rnn = model.fit(texts_train, y_train, epochs=10, batch_size=60, validation_split=0.2)

    epochs = 5
    #model.compile(loss='mean_absolute_error',optimizer='adam',metrics=['accuracy'])
    #binary_crossentropy
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    print(short_model_summary)

    history = model.fit(X_train, y_train, 
                        epochs = epochs, verbose = 1, validation_split = 0.2, 
                        batch_size = 32)
    final_fit_time = time.time()
    time_fit = final_fit_time - start_fit_time

    pred = model.predict_classes(X_test)
    final_test_time = time.time()
    time_test = final_test_time - final_fit_time
    acc = model.evaluate(X_test, y_test)

    #print("Test accuracy is {1:.2f} % ".format(acc[1]*100))
    print(confusion_matrix(pred, y_test))

    statistics = "" 
    statistics += "<div>Confusion matrix</div>"  
    dfStats = pd.DataFrame(confusion_matrix(pred, y_test))
    statistics += "<div>" + dfStats.to_html() + "</div>"
    statistics += "<div> accuracy = "+str(acc[1]*100) + " % </div>"
    statistics += "<div> time_fit = "+ str(time_fit) + " sg. </div>" 
    statistics += "<div> time_test = "+ str(time_test) + " sg. </div>" 

    return tokenizer, model, statistics, short_model_summary

def clasiffierLSTM(cbp, tokenizer, model):

    start_pred_time = time.time()
    seq_texts = list(getKmers(cbp.querry_seq,6))
    seq_texts = ' '.join(seq_texts)

    seq = list()
    seq.append(seq_texts)

    # encode document
    X_seq = tokenizer.texts_to_sequences(seq)

    y_pred = model.predict_classes(X_seq)
    final_pred_time = time.time()
    print("Predictions x_test: ", str(y_pred[0]))
    print("Time Prediction: " + str(final_pred_time-start_pred_time) + " sg.")

    if y_pred[0]==1:
        return "Identificate pathogen: SARS-CoV-2 | Severe acute respiratory syndrome coronavirus 2 isolate Wuhan-Hu-1(complete genome)", str(final_pred_time-start_pred_time) + " sg."
    else:
       return "No identificate pathogen SARS-CoV-2", str(final_pred_time-start_pred_time) + " sg."
    