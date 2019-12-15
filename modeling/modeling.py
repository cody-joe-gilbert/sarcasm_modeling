'''
Statistical NLP: Final Project
Cody Gilbert

This module iterates over the Keras models defined in text_models.py
and iterates over all considered model hyperparameters to find the model 
with the highest accuracy metrics.

'''

import sys
import os
import pandas as pd
import numpy as np
import multiprocessing as mp
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import logging
import logging.handlers
logger = logging.getLogger('__name__')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - \n\t%(message)s')
file_handler = logging.handlers.RotatingFileHandler('modeling.log',
                                                maxBytes=1e6,
                                                backupCount=3)
logger.addHandler(file_handler)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)

import text_models

def iterate_params(run_case,
                   model_params,):
    '''
	Dispatches a model fitting execution for a given combination
	of model hyperparameters.
	Originally used to dispatch to multiprocessing pool, however
	time constraints required sequential execution.
    '''
    
    model_name = model_params['model_name']
    epochs = model_params['epochs']
    cell_units = model_params['cell_units']
    msg = (f'Run Case: {run_case}'
            f' model_name: {model_name}'
            f' epochs: {epochs}'
            f' cell_units: {cell_units}')
    logger.debug(msg)
    
    if model_params["model_name"] == 'NSObsceneLSTM':
        model = text_models.NSObsceneLSTM
    elif model_params["model_name"] == 'ObsceneLSTM':
        model = text_models.ObsceneLSTM
    elif model_params["model_name"] == 'NoSeqLSTM':
        model = text_models.NoSeqLSTM
    elif model_params["model_name"] == 'PlainLSTM':
        model = text_models.PlainLSTM
    else:
        model = None
    model_obj, results = model(model_params)
    # Read the selected number of lines into a temp file
    loss = results[0]
    accuracy = results[1]
    precision = results[2]
    recall = results[3]
    auc = results[4]
    msg = (f'Finished Run Case: {run_case}'
            f' Accuracy: {accuracy}'
            f' Loss: {loss}'
            f' Precision: {precision}'
            f' Recall: {recall}'
            f' AUC: {auc}')
    logger.debug(msg)
    
    return model_params, run_case, list(results)

vocab_size = 5000
data_size = 10000
featurize = False
max_review_length = 200
embedding_vector_length = 200
test_fraction = 0.33
valid_fraction = 0.25
features_file = 'features.csv'
raw_reviews_file = 'train-balanced-sarcasm.csv'
results_file = os.path.join(os.getcwd(), 'results.csv')

logger.debug('Loading tokenizer')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
logger.debug('Loading model')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
logger.debug('Pushing model to eval')
model.eval()

obscene_words_file = 'obscene_corpus.txt'
obscene_data = pd.read_csv(obscene_words_file, index_col=None, header=None, names=['words'])
obscene_words = list(obscene_data['words'])

msg = (f'Fitting models with:\n'
        f'\t Vocab Size: {vocab_size}\n'
        f'\t Max Comment Len: {max_review_length}\n'
        f'\t Embedding Vector Len: {embedding_vector_length}\n'
        f'\t Test Set Frac: {test_fraction}\n'
        f'\t Validation Set Frac: {valid_fraction}\n'
        f'\t Features Corpus: {features_file}')

logger.debug(msg)

# Define feature-generating functions for dataframe mapping
def clean_text(value):
    if value is None or type(value) is type(0.2):
        return None 
    clean_text = []
    for c in value:
        if c.isalpha() or c == ' ':
            clean_text.append(c.lower())
    return ''.join(clean_text)

def comment_text_feature(comment):
    clean_comment = clean_text(comment)
    if clean_comment is None or clean_comment == '':
        return None
    return clean_comment
    
def comment_obscene_feature(comment):
    clean_comment = clean_text(comment)
    if clean_comment is None:
        return None
    tokens = clean_comment.split()
    for word in tokens:
        if word in obscene_words:
            return 1
    return 0

def comment_nonseq_feature(comment):
    if comment is None or comment == '':
        return None
    parsed_text = ['[CLS] ']
    for c in comment:
        if c.isalpha() or c == ' ':
            parsed_text.append(c)
        elif c == '.' or c == '?' or c == '!':
            parsed_text.append(c + ' [SEP]')
        else:
            continue
    parsed_text = ''.join(parsed_text)
    try:
        input_ids = torch.tensor(tokenizer.encode(parsed_text, add_special_tokens=True)).unsqueeze(0)
        if len(input_ids) > 512:
            logger.debug('Excessive length on %s', parsed_text)
            return None
        outputs = model(input_ids, masked_lm_labels=input_ids)
    except RuntimeError:
        logger.debug('Model error on %s', parsed_text)
        return None
    return outputs[0].item()

def comment_bert_feature(comment):
    if comment is None or comment == '':
        return None
    parsed_text = ['[CLS] ']
    for c in comment:
        if c.isalpha() or c == ' ':
            parsed_text.append(c)
        elif c == '.' or c == '?' or c == '!':
            parsed_text.append(c + ' [SEP]')
        else:
            continue
    parsed_text = ''.join(parsed_text)
    try:
        input_ids = torch.tensor(tokenizer.encode(parsed_text, add_special_tokens=True)).unsqueeze(0)
        if len(input_ids) > 512:
            logger.debug('Excessive length on %s', parsed_text)
            return None
    except RuntimeError:
        logger.debug('Model error on %s', parsed_text)
        return None
    return input_ids

# Load in and featurize the data
if os.path.exists(features_file) and featurize:
    logger.debug('Importing features')
    features_df = pd.read_csv(features_file, index_col=None, nrows=data_size)
    features_df = features_df.dropna()
else:
    logger.debug('Importing and featurizing data')
    input_data = pd.read_csv(raw_reviews_file, index_col=None, nrows=data_size)
    logger.debug(f'Input data length: {len(input_data)}')
    
    features_df = pd.DataFrame([])
    features_df['label'] = input_data['label']
    features_df['comment_text_feature'] = input_data['comment'].apply(comment_text_feature)
    features_df['comment_obscene_feature'] = input_data['comment'].apply(comment_obscene_feature)
    features_df['comment_nonseq_feature'] = input_data['comment'].apply(comment_nonseq_feature)
	features_df['comment_bert_feature'] = input_data['comment'].apply(comment_bert_feature)
    
    features_df = features_df.dropna()
    logger.debug(f'Feature data length: {len(features_df)}')
    features_df.to_csv(features_file, index=False)

comments = list(features_df['comment_text_feature'])
obscene_comment = features_df['comment_obscene_feature'].to_numpy()
nonseq_feature = features_df['comment_nonseq_feature'].to_numpy()
bert_feature = features_df['comment_bert_feature'].to_numpy()
labels = features_df['label'].to_numpy()

# Encode the review words
logger.debug('Encoding words')
encoded_comments = [one_hot(r, vocab_size) for r in comments]
padded_comments = pad_sequences(encoded_comments, maxlen=max_review_length, padding='post')

res_dict = {'Run Case': [],
            'Model Name': [],
            'Loss': [],
            'Accuracy': [],
            'Precision': [],
            'Recall': [],
            'AUC': [],
            'Embedding Length': [],
            'Epochs': [],
            'Cell Units': [],
            'Dense Units': [],
            }

# iterate over the hyperparameter space to train models
run_case = 0
for model_name in ['NSObsceneLSTM', 'ObsceneLSTM', 'NoSeqLSTM', 'PlainLSTM']:
    for epochs in range(1, 6): 
        for cell_units in [300, 400]:
            for embedding_vector_length in [200, 300]:
                for dense_units in [20, 30, 40]:
                    model_params = {
                                'model_name': model_name,
                                'word_features': padded_comments,
                                'obscene_features': obscene_comment,
                                'nonseq_features': nonseq_feature,
                                'labels': labels,
                                'cell_units': cell_units,
                                'epochs': epochs,
                                'embedding_vector_length': embedding_vector_length,
                                'vocab_size': vocab_size,
                                'dense_units': dense_units,
                            }
                    run_case += 1
                    model_params, run_case, results = iterate_params(run_case, model_params)
                    res_dict['Run Case'].append(run_case)
                    res_dict['Model Name'].append(model_params['model_name'])
                    res_dict['Epochs'].append(model_params['epochs'])
                    res_dict['Embedding Length'].append(model_params['embedding_vector_length'])
                    res_dict['Cell Units'].append(model_params['cell_units'])
                    res_dict['Dense Units'].append(model_params['dense_units'])
                    res_dict['Loss'].append(results[0])
                    res_dict['Accuracy'].append(results[1])
                    res_dict['Precision'].append(results[2])
                    res_dict['Recall'].append(results[3])
                    res_dict['AUC'].append(results[4])
                    # Save modeling results to file
                    results_frame = pd.DataFrame(res_dict)
                    results_frame.to_csv(results_file, index=False)
