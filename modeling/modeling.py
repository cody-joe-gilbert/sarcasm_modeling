import sys
import os
import pandas as pd
import numpy as np
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

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

vocab_size = 5000
max_review_length = 200
embedding_vector_length = 32
test_fraction = 0.33
valid_fraction = 0.25
reviews_file = 'train-balanced-sarcasm_cleaned.csv'
raw_reviews_file = 'train-balanced-sarcasm.csv'

msg = (f'Fitting models with:\n'
        f'\t Vocab Size: {vocab_size}\n'
        f'\t Max Comment Len: {max_review_length}\n'
        f'\t Embedding Vector Len: {embedding_vector_length}\n'
        f'\t Test Set Frac: {test_fraction}\n'
        f'\t Validation Set Frac: {valid_fraction}\n'
        f'\t Comments Corpus: {reviews_file}')

logger.debug(msg)

def clean_text(value):
    if value is None or type(value) is type(0.2):
        return value 
    clean_text = []
    for c in value:
        if c.isalpha() or c == ' ':
            clean_text.append(c.lower())
    return ''.join(clean_text)


if os.path.exists(reviews_file):
    logger.debug('Importing data')
    text = pd.read_csv(reviews_file, index_col=None, nrows=10000)
    reals = text['comment'].notna()
    text = text.loc[reals, :]
else:
    logger.debug('Importing and cleaning data')
    text = pd.read_csv(raw_reviews_file, index_col=None)
    reals = text['comment'].notna()
    text = text.loc[reals, ['label', 'comment']]
    text['comment'] = text['comment'].apply(clean_text)
    reals = text['comment'].notna()
    text = text.loc[reals, ['label', 'comment']]
    text.to_csv(reviews_file, index=False)

reviews = list(text['comment'])
labels = text['label'].to_numpy()

# Encode the review words
logger.debug('Encoding words')
encoded_reviews = [one_hot(r, vocab_size) for r in reviews]
padded_reviews = pad_sequences(encoded_reviews, maxlen=max_review_length, padding='post')

# Split into training, test, and validation sets
logger.debug('Splitting sets')
train_x, test_x, train_y, test_y = train_test_split(padded_reviews, labels, test_size=test_fraction)
dataset = train_test_split(train_x, train_y, test_size=valid_fraction)

msg = (f'Dataset metrics: \n'
        f'\t Train Set Len: {len(train_x)}\n'
        f'\t Validation Set Len: {len(dataset[1])}\n'
        f'\t Test Set Len: {len(test_x)}\n'
        f'\t Train Set Positives: {np.sum(train_y)}\n'
        f'\t Validation Set Positives: {np.sum(dataset[3])}\n'
        f'\t Test Set Positives: {np.sum(test_y)}'
        )
logger.debug(msg)

###########################################
###########################################

kwargs = {
    'cell_units': 50,
    'epochs': 20,
    'embedding_vector_length': embedding_vector_length,
    'vocab_size': vocab_size,
}
model = text_models.fit_BidirectionalLSTM(dataset, **kwargs)
logger.debug('Evaluate on test data')
results = model.evaluate(test_x, test_y, batch_size=128)
logger.debug('test metrics: %s', results)





