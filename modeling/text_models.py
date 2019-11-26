import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, SimpleRNN, LSTM, Embedding, Dropout
import keras.layers as layers

import logging
import logging.handlers
logger = logging.getLogger('__name__')
logger.propgate = True


def fit_SimpleRNN(datasets, embedding_vector_length, vocab_size, cell_units, epochs):
    '''
    Fit RNN to data train_x, train_y 
    
    Args:
        train_x (array): input sequence samples for training 
        train_y (list): next step in sequence targets
        cell_units (int): number of hidden cells, or output vector size 
        epochs (int): number of training epochs   
    '''
    model_name = 'simplernn.mod'
    msg = (f'Fitting fit_SimpleRNN with:\n'
            f'\t Vocab Size: {vocab_size}\n'
            f'\t Embedding Vector Len: {embedding_vector_length}\n'
            f'\t Cell Units: {cell_units}\n'
            f'\t Epochs: {epochs}\n'
            f'\t Model File: {model_name}'
            )
    logger.debug(msg)
    
    train_x, valid_x, train_y, valid_y = datasets
    # initialize model
    model = Sequential() 
    model.add(layers.Embedding(vocab_size, embedding_vector_length, input_length=len(train_x[0])))
    model.add(layers.Dropout(0.2))
    model.add(layers.SimpleRNN(cell_units))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', 
                  optimizer='adam',
                  metrics=[keras.metrics.BinaryAccuracy(),
                            keras.metrics.Precision(),
                            keras.metrics.Recall(),
                            keras.metrics.AUC()])
    model.fit(train_x, train_y, validation_data=(valid_x, valid_y), epochs=epochs, batch_size=128, verbose=1)
    model.save(model_name)
    return model
    
def fit_SimpleLSTM(datasets, embedding_vector_length, vocab_size, cell_units, epochs):
    '''
    Fit LSTM to data train_x, train_y 
    
    Args:
        train_x (array): input sequence samples for training 
        train_y (list): next step in sequence targets
        cell_units (int): number of hidden cells, or output vector size 
        epochs (int): number of training epochs   
    '''
    model_name = 'Simplelstm.mod'
    msg = (f'Fitting fit_SimpleRNN with:\n'
            f'\t Vocab Size: {vocab_size}\n'
            f'\t Embedding Vector Len: {embedding_vector_length}\n'
            f'\t Cell Units: {cell_units}\n'
            f'\t Epochs: {epochs}\n'
            f'\t Model File: {model_name}'
            )
    logger.debug(msg)
    train_x, valid_x, train_y, valid_y = datasets
    # initialize model
    model = Sequential() 
    model.add(layers.Embedding(vocab_size, embedding_vector_length, input_length=len(train_x[0])))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(cell_units))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', 
                  optimizer='adam',
                  metrics=[keras.metrics.BinaryAccuracy(),
                            keras.metrics.Precision(),
                            keras.metrics.Recall(),
                            keras.metrics.AUC()])
    model.fit(train_x, train_y, validation_data=(valid_x, valid_y), epochs=epochs, batch_size=128, verbose=1)
    model.save(model_name)
    return model
    
def fit_BidirectionalLSTM(datasets, embedding_vector_length, vocab_size, cell_units, epochs):
    '''
    Fit LSTM 
    
    Args:
        train_x (array): input sequence samples for training 
        train_y (list): next step in sequence targets
        cell_units (int): number of hidden cells, or output vector size 
        epochs (int): number of training epochs   
    '''
    model_name = 'BidirectionalLSTM'
    msg = (f'Fitting {model_name} with:\n'
            f'\t Vocab Size: {vocab_size}\n'
            f'\t Embedding Vector Len: {embedding_vector_length}\n'
            f'\t Cell Units: {cell_units}\n'
            f'\t Epochs: {epochs}\n'
            f'\t Model File: {model_name}.mod'
            )
    logger.debug(msg)
    train_x, valid_x, train_y, valid_y = datasets
    # initialize model
    model = Sequential() 
    model.add(layers.Embedding(vocab_size, embedding_vector_length, input_length=len(train_x[0])))
    model.add(layers.Bidirectional(layers.LSTM(cell_units*2, return_sequences=True)))
    model.add(layers.Bidirectional(layers.LSTM(cell_units)))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', 
                  optimizer='adam',
                  metrics=[keras.metrics.BinaryAccuracy(),
                            keras.metrics.Precision(),
                            keras.metrics.Recall(),
                            keras.metrics.AUC()])
    model.fit(train_x, train_y, validation_data=(valid_x, valid_y), epochs=epochs, batch_size=128, verbose=1)
    model.save(model_name + '.mod')
    return model