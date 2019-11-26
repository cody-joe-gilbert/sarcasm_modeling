import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import pandas as pd
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.classify import apply_features
from nltk.util import ngrams

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

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


vocab_size = 5000
max_review_length = 200
embedding_vector_length = 32
test_fraction = 0.33
valid_fraction = 0.25
reviews_file = 'train-balanced-sarcasm_cleaned.csv'

msg = (f'Fitting models with:\n'
        f'\t Vocab Size: {vocab_size}\n'
        f'\t Max Comment Len: {max_review_length}\n'
        f'\t Embedding Vector Len: {embedding_vector_length}\n'
        f'\t Test Set Frac: {test_fraction}\n'
        f'\t Validation Set Frac: {valid_fraction}\n'
        f'\t Comments Corpus: {reviews_file}')

logger.debug('Importing data')
text = pd.read_csv(reviews_file, index_col=None, nrows=10000)
reals = text['comment'].notna()
text = text.loc[reals, :]


train, test = train_test_split(text, test_size=test_fraction)

# Tokenize, Remove Stop words, and Lemmatize
def clean_words(text):
    if type(text) is type(''):
        clean_text = []
        sentences = text.split('.')
        for sentence in sentences:
            clean_sentence = []
            tokens = word_tokenize(sentence)
            for w in tokens:
                if w not in stop_words:
                    clean_sentence.append(lemmatizer.lemmatize(w))
            if len(clean_sentence) > 0:
                clean_text += clean_sentence
    return(clean_text)

def text_features(row):
    text = clean_words(row['comment'])
    label = row['label']
    sentences = [sentence for sentence in comment.split(".") if sentence != ""]
    tokens = []
    for sentence in sentences:
        tokens += [token for token in sentences.split(" ") if token != ""]
    return (list(ngrams(tokens, 3)), label)

fdist = nltk.FreqDist()
for _, row in text.iterrows():
    for word in clean_words(row['comment']):
        fdist[word.lower()] += 1

word_features = list(fdist)[:vocab_size]

def comment_word_features(row_iter):
    _, row = row_iter
    text = clean_words(row['comment'])
    label = row['label']
    comment_words = set(text)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in comment_words)
    return (features, label)

train_set = [comment_word_features(x) for x in train.iterrows()]
test_set = [comment_word_features(x) for x in test.iterrows()]
logger.debug('Training classifier')
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))
