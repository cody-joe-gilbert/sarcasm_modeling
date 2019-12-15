'''
Statistical NLP: Final Project
Cody Gilbert

This module defines a series of Keras models used by the modeling.py 
module for iterating over a grid of modeling parameters.

'''
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, SimpleRNN, LSTM, Embedding, Dropout
import keras.layers as layers
from sklearn.model_selection import train_test_split
import logging
import logging.handlers
logger = logging.getLogger('__name__')
logger.propgate = True

def split_sets(features, 
                labels,
                dev_frac=0.25,
                test_frac=0.25,
                seed=42
                ):
    '''
    Performs training, test, and dev set splitting for a given single set of data
    '''
    logger.debug('Splitting sets')
    train_f, test_f, train_l, test_l = train_test_split(features, 
                                                        labels, 
                                                        test_size=test_frac,
                                                        random_state=seed)
    train_f, valid_f, train_l, valid_l = train_test_split(train_f, 
                                                          train_l, 
                                                          test_size=dev_frac,
                                                          random_state=seed)
    
    msg = (f'Dataset Metrics: \n'
        f'\t Train Set Len: {len(train_f)}\n'
        f'\t Validation Set Len: {len(valid_f)}\n'
        f'\t Test Set Len: {len(test_f)}\n'
        f'\t Train Set Positives: {np.sum(train_l)}\n'
        f'\t Validation Set Positives: {np.sum(valid_l)}\n'
        f'\t Test Set Positives: {np.sum(test_l)}'
        )
    logger.debug(msg)
    datasets = {
        'train_features': train_f,
        'train_labels': train_l,
        'valid_features': valid_f,
        'valid_labels': valid_l,
        'test_features': test_f,
        'test_labels': test_l,
    }
    return datasets

def fit_SimpleRNN(features, 
                  labels, 
                  embedding_vector_length, 
                  vocab_size, 
                  cell_units, 
                  epochs):
    '''
    Fits a simple RNN layer using Keras and evaluates validation set metrics
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
    datasets = split_sets(features, labels)
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
    model.fit(datasets['train_features'], 
              datasets['train_labels'], 
              validation_data=(datasets['valid_features'], datasets['valid_labels']), 
              epochs=epochs, 
              batch_size=128, 
              verbose=1)
    model.save(model_name)
    logger.debug('Evaluate on test data')
    results = model.evaluate(datasets['test_features'], datasets['test_labels'], batch_size=128)
    logger.debug('Validation Set (%s epochs) metrics: %s', epochs, results)
    return model

def fit_SimpleLSTM(features, 
                  labels, 
                  embedding_vector_length, 
                  vocab_size, 
                  cell_units, 
                  epochs):
    '''
    Fits a simple LSTM layer using Keras and evaluates validation set metrics
    '''
    model_name = 'SimpleLSTM'
    msg = (f'Fitting {model_name} with:\n'
            f'\t Vocab Size: {vocab_size}\n'
            f'\t Embedding Vector Len: {embedding_vector_length}\n'
            f'\t Cell Units: {cell_units}\n'
            f'\t Epochs: {epochs}\n'
            f'\t Model File: {model_name}'
            )
    logger.debug(msg)
    datasets = split_sets(features, labels)
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
    model.fit(datasets['train_features'], 
              datasets['train_labels'], 
              validation_data=(datasets['valid_features'], datasets['valid_labels']), 
              epochs=epochs, 
              batch_size=128, 
              verbose=1)
    model.save(model_name + '.mod')
    logger.debug('Evaluate on test data')
    results = model.evaluate(datasets['test_features'], datasets['test_labels'], batch_size=128)
    logger.debug('Test Set metrics: %s', results)
    return model

def fit_BidirectionalLSTM(features, 
                        labels, 
                        embedding_vector_length, 
                        vocab_size, 
                        cell_units, 
                        epochs):
    '''
    Fits a Bidirectional LSTM layer using Keras and evaluates validation set metrics
    '''
    model_name = 'BidirectionalLSTM'
    msg = (f'Fitting {model_name} with:\n'
            f'\t Vocab Size: {vocab_size}\n'
            f'\t Embedding Vector Len: {embedding_vector_length}\n'
            f'\t Cell Units: {cell_units}\n'
            f'\t Epochs: {epochs}\n'
            f'\t Model File: {model_name}'
            )
    logger.debug(msg)
    datasets = split_sets(features, labels)
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
    model.save(model_name + '.mod')
    logger.debug('Evaluate on test data')
    results = model.evaluate(datasets['test_features'], datasets['test_labels'], batch_size=128)
    logger.debug('Test Set metrics: %s', results)
    return model

def fit_ObsceneLSTM(word_features,
                    obscene_features,
                    labels, 
                    embedding_vector_length, 
                    vocab_size, 
                    cell_units, 
                    epochs):
    '''
    Fits a modified LSTM model with an additional OI feature and evaluates validation set metrics
    '''
    model_name = 'ObsceneLSTM'
    msg = (f'Fitting {model_name} with:\n'
            f'\t Vocab Size: {vocab_size}\n'
            f'\t Embedding Vector Len: {embedding_vector_length}\n'
            f'\t Cell Units: {cell_units}\n'
            f'\t Epochs: {epochs}\n'
            f'\t Model File: {model_name}'
            )
    logger.debug(msg)
    # Split the word and categorical features together
    (train_wf, test_wf,
    train_of, test_of,
    train_l, test_l) = train_test_split(word_features,
                                        obscene_features,
                                        labels, 
                                        test_size=0.25,
                                        random_state=42)
    (train_wf, valid_wf,
    train_of, valid_of,
    train_l, valid_l) = train_test_split(train_wf,
                                        train_of,
                                        train_l, 
                                        test_size=0.25,
                                        random_state=42)
    datasets = {
        'train_word_features': train_wf,
        'train_obscene_features': train_of,
        'train_labels': train_l,
        'valid_word_features': valid_wf,
        'valid_obscene_features': valid_of,
        'valid_labels': valid_l,
        'test_word_features': test_wf,
        'test_obscene_features': test_of,
        'test_labels': test_l,
    }
    # initialize model
    word_input = keras.Input(shape=(None,), name='words')
    obscene_input = keras.Input(shape=(1, ), name='obscenes')
    
    word_layer = layers.Embedding(vocab_size, embedding_vector_length)(word_input)
    word_lstm = layers.LSTM(cell_units)(word_layer)
    full_features = layers.concatenate([word_lstm, obscene_input])
    pred_layer = layers.Dense(1, activation='sigmoid', name='class')(full_features)
    model = keras.Model(inputs=[word_input, obscene_input],
                        outputs=pred_layer)
    model.compile(loss='binary_crossentropy', 
                  optimizer='adam',
                  metrics=[keras.metrics.BinaryAccuracy(),
                            keras.metrics.Precision(),
                            keras.metrics.Recall(),
                            keras.metrics.AUC()])
    model.fit({'words': datasets['train_word_features'], 
                'obscenes': datasets['train_obscene_features']},
              {'class': datasets['train_labels']},
              epochs=epochs,
              batch_size=128,
              verbose=1
              )

    model.save(model_name + '.mod')
    logger.debug('Evaluate on validation data')
    results = model.evaluate({'words': datasets['valid_word_features'], 
                              'obscenes': datasets['valid_obscene_features']}, 
                              {'class': datasets['valid_labels']}, 
                              batch_size=128)
    logger.debug('Test Set metrics: %s', results)
    return model

def NSObsceneLSTM(model_params):
    '''
    Fits the full hybrid model with NS and OI features and evaluates validation set metrics
    '''
    
    # Split the word and categorical features together
    (train_wf, test_wf,
    train_of, test_of,
    train_nsf, test_nsf,
    train_l, test_l) = train_test_split(model_params['word_features'],
                                        model_params['obscene_features'],
                                        model_params['nonseq_features'],
                                        model_params['labels'],
                                        test_size=0.25,
                                        random_state=42)
    (train_wf, valid_wf,
    train_of, valid_of,
    train_nsf, valid_nsf,
    train_l, valid_l) = train_test_split(train_wf,
                                        train_of,
                                        train_nsf,
                                        train_l,
                                        test_size=0.25,
                                        random_state=42)
    datasets = {
        'train_word_features': train_wf,
        'train_obscene_features': train_of,
        'train_noseq_features': train_nsf,
        'train_labels': train_l,
        'valid_word_features': valid_wf,
        'valid_obscene_features': valid_of,
        'valid_noseq_features': valid_nsf,
        'valid_labels': valid_l,
        'test_word_features': test_wf,
        'test_obscene_features': test_of,
        'test_noseq_features': test_nsf,
        'test_labels': test_l,
    }
    # initialize model
    word_input = keras.Input(shape=(None,), name='words')
    obscene_input = keras.Input(shape=(1, ), name='obscenes')
    noseq_input = keras.Input(shape=(1, ), name='noseqs')
    word_layer = layers.Embedding(model_params['vocab_size'], model_params['embedding_vector_length'])(word_input)
    word_lstm = layers.Bidirectional(layers.LSTM(model_params['cell_units'], return_sequences=True))(word_layer)
    do_1 = layers.Dropout(0.2)(word_lstm)
    word_lstm2 = layers.Bidirectional(layers.LSTM(model_params['cell_units']))(do_1)
    do_2 = layers.Dropout(0.2)(word_lstm2)
    full_features = layers.concatenate([do_2, obscene_input, noseq_input])
    dense_layer = layers.Dense(model_params['dense_units'], activation='relu', name='dense1')(full_features)
    do_3 = layers.Dropout(0.2)(dense_layer)
    pred_layer = layers.Dense(1, activation='sigmoid', name='class')(do_3)
    model = keras.Model(inputs=[word_input, obscene_input, noseq_input],
                        outputs=pred_layer)
    model.compile(loss='binary_crossentropy', 
                  optimizer='adam',
                  metrics=[keras.metrics.BinaryAccuracy(),
                            keras.metrics.Precision(),
                            keras.metrics.Recall(),
                            keras.metrics.AUC()])
    model.fit({'words': datasets['train_word_features'], 
               'obscenes': datasets['train_obscene_features'],
               'noseqs': datasets['train_noseq_features'],},
              {'class': datasets['train_labels']},
              epochs=model_params['epochs'],
              batch_size=128,
              verbose=1
              )

    model.save(model_params['model_name'] + '.mod')
    logger.debug('Evaluate on validation data')
    results = model.evaluate({'words': datasets['valid_word_features'], 
                              'obscenes': datasets['valid_obscene_features'],
                              'noseqs': datasets['valid_noseq_features'],}, 
                              {'class': datasets['valid_labels']}, 
                              batch_size=128)
    return model, results

def ObsceneLSTM(model_params):
    '''
    Fits the hybrid model with OI features and evaluates validation set metrics
    '''

    # Split the word and categorical features together
    (train_wf, test_wf,
    train_of, test_of,
    train_nsf, test_nsf,
    train_l, test_l) = train_test_split(model_params['word_features'],
                                        model_params['obscene_features'],
                                        model_params['nonseq_features'],
                                        model_params['labels'],
                                        test_size=0.25,
                                        random_state=42)
    (train_wf, valid_wf,
    train_of, valid_of,
    train_nsf, valid_nsf,
    train_l, valid_l) = train_test_split(train_wf,
                                        train_of,
                                        train_nsf,
                                        train_l,
                                        test_size=0.25,
                                        random_state=42)
    datasets = {
        'train_word_features': train_wf,
        'train_obscene_features': train_of,
        'train_noseq_features': train_nsf,
        'train_labels': train_l,
        'valid_word_features': valid_wf,
        'valid_obscene_features': valid_of,
        'valid_noseq_features': valid_nsf,
        'valid_labels': valid_l,
        'test_word_features': test_wf,
        'test_obscene_features': test_of,
        'test_noseq_features': test_nsf,
        'test_labels': test_l,
    }
    # initialize model
    word_input = keras.Input(shape=(None,), name='words')
    obscene_input = keras.Input(shape=(1, ), name='obscenes')
    
    word_layer = layers.Embedding(model_params['vocab_size'], model_params['embedding_vector_length'])(word_input)
    word_lstm = layers.Bidirectional(layers.LSTM(model_params['cell_units'], return_sequences=True))(word_layer)
    do_1 = layers.Dropout(0.2)(word_lstm)
    word_lstm2 = layers.Bidirectional(layers.LSTM(model_params['cell_units']))(do_1)
    do_2 = layers.Dropout(0.2)(word_lstm2)
    full_features = layers.concatenate([do_2, obscene_input])
    dense_layer = layers.Dense(model_params['dense_units'], activation='relu', name='dense1')(full_features)
    do_3 = layers.Dropout(0.2)(dense_layer)
    pred_layer = layers.Dense(1, activation='sigmoid', name='class')(do_3)
    model = keras.Model(inputs=[word_input, obscene_input],
                        outputs=pred_layer)
    model.compile(loss='binary_crossentropy', 
                  optimizer='adam',
                  metrics=[keras.metrics.BinaryAccuracy(),
                            keras.metrics.Precision(),
                            keras.metrics.Recall(),
                            keras.metrics.AUC()])
    model.fit({'words': datasets['train_word_features'], 
               'obscenes': datasets['train_obscene_features'],},
              {'class': datasets['train_labels']},
              epochs=model_params['epochs'],
              batch_size=128,
              verbose=1
              )

    model.save(model_params['model_name'] + '.mod')
    logger.debug('Evaluate on validation data')
    results = model.evaluate({'words': datasets['valid_word_features'], 
                              'obscenes': datasets['valid_obscene_features'],}, 
                              {'class': datasets['valid_labels']}, 
                              batch_size=128)
    return model, results

def NoSeqLSTM(model_params):
    '''
    Fits the hybrid model with NS features and evaluates validation set metrics
    '''

    # Split the word and categorical features together
    (train_wf, test_wf,
    train_of, test_of,
    train_nsf, test_nsf,
    train_l, test_l) = train_test_split(model_params['word_features'],
                                        model_params['obscene_features'],
                                        model_params['nonseq_features'],
                                        model_params['labels'],
                                        test_size=0.25,
                                        random_state=42)
    (train_wf, valid_wf,
    train_of, valid_of,
    train_nsf, valid_nsf,
    train_l, valid_l) = train_test_split(train_wf,
                                        train_of,
                                        train_nsf,
                                        train_l,
                                        test_size=0.25,
                                        random_state=42)
    datasets = {
        'train_word_features': train_wf,
        'train_obscene_features': train_of,
        'train_noseq_features': train_nsf,
        'train_labels': train_l,
        'valid_word_features': valid_wf,
        'valid_obscene_features': valid_of,
        'valid_noseq_features': valid_nsf,
        'valid_labels': valid_l,
        'test_word_features': test_wf,
        'test_obscene_features': test_of,
        'test_noseq_features': test_nsf,
        'test_labels': test_l,
    }
    # initialize model
    word_input = keras.Input(shape=(None,), name='words')
    noseq_input = keras.Input(shape=(1, ), name='noseqs')
    
    word_layer = layers.Embedding(model_params['vocab_size'], model_params['embedding_vector_length'])(word_input)
    word_lstm = layers.Bidirectional(layers.LSTM(model_params['cell_units'], return_sequences=True))(word_layer)
    do_1 = layers.Dropout(0.2)(word_lstm)
    word_lstm2 = layers.Bidirectional(layers.LSTM(model_params['cell_units']))(do_1)
    do_2 = layers.Dropout(0.2)(word_lstm2)
    full_features = layers.concatenate([do_2, noseq_input])
    dense_layer = layers.Dense(model_params['dense_units'], activation='relu', name='dense1')(full_features)
    do_3 = layers.Dropout(0.2)(dense_layer)
    pred_layer = layers.Dense(1, activation='sigmoid', name='class')(do_3)
    model = keras.Model(inputs=[word_input, noseq_input],
                        outputs=pred_layer)
    model.compile(loss='binary_crossentropy', 
                  optimizer='adam',
                  metrics=[keras.metrics.BinaryAccuracy(),
                            keras.metrics.Precision(),
                            keras.metrics.Recall(),
                            keras.metrics.AUC()])
    model.fit({'words': datasets['train_word_features'],
               'noseqs': datasets['train_noseq_features'],},
              {'class': datasets['train_labels']},
              epochs=model_params['epochs'],
              batch_size=128,
              verbose=1
              )

    model.save(model_params['model_name'] + '.mod')
    logger.debug('Evaluate on validation data')
    results = model.evaluate({'words': datasets['valid_word_features'],
                              'noseqs': datasets['valid_noseq_features'],}, 
                              {'class': datasets['valid_labels']}, 
                              batch_size=128)
    return model, results

def PlainLSTM(model_params):
    '''
    Fits the hybrid model with only text features and evaluates validation set metrics
    '''

    # Split the word and categorical features together
    (train_wf, test_wf,
    train_of, test_of,
    train_nsf, test_nsf,
    train_l, test_l) = train_test_split(model_params['word_features'],
                                        model_params['obscene_features'],
                                        model_params['nonseq_features'],
                                        model_params['labels'],
                                        test_size=0.25,
                                        random_state=42)
    (train_wf, valid_wf,
    train_of, valid_of,
    train_nsf, valid_nsf,
    train_l, valid_l) = train_test_split(train_wf,
                                        train_of,
                                        train_nsf,
                                        train_l,
                                        test_size=0.25,
                                        random_state=42)
    datasets = {
        'train_word_features': train_wf,
        'train_obscene_features': train_of,
        'train_noseq_features': train_nsf,
        'train_labels': train_l,
        'valid_word_features': valid_wf,
        'valid_obscene_features': valid_of,
        'valid_noseq_features': valid_nsf,
        'valid_labels': valid_l,
        'test_word_features': test_wf,
        'test_obscene_features': test_of,
        'test_noseq_features': test_nsf,
        'test_labels': test_l,
    }
    # initialize model
    word_input = keras.Input(shape=(None,), name='words')
    
    word_layer = layers.Embedding(model_params['vocab_size'], model_params['embedding_vector_length'])(word_input)
    word_lstm = layers.Bidirectional(layers.LSTM(model_params['cell_units'], return_sequences=True))(word_layer)
    do_1 = layers.Dropout(0.2)(word_lstm)
    word_lstm2 = layers.Bidirectional(layers.LSTM(model_params['cell_units']))(do_1)
    do_2 = layers.Dropout(0.2)(word_lstm2)
    dense_layer = layers.Dense(model_params['dense_units'], activation='relu', name='dense1')(do_2)
    do_3 = layers.Dropout(0.2)(dense_layer)
    pred_layer = layers.Dense(1, activation='sigmoid', name='class')(do_3)
    model = keras.Model(inputs=[word_input, noseq_input],
                        outputs=pred_layer)
    model.compile(loss='binary_crossentropy', 
                  optimizer='adam',
                  metrics=[keras.metrics.BinaryAccuracy(),
                            keras.metrics.Precision(),
                            keras.metrics.Recall(),
                            keras.metrics.AUC()])
    model.fit({'words': datasets['train_word_features'],},
              {'class': datasets['train_labels']},
              epochs=model_params['epochs'],
              batch_size=128,
              verbose=1
              )

    model.save(model_params['model_name'] + '.mod')
    logger.debug('Evaluate on validation data')
    results = model.evaluate({'words': datasets['valid_word_features'],}, 
                              {'class': datasets['valid_labels']}, 
                              batch_size=128)
    return model, results

