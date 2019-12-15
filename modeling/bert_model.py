'''
Statistical NLP: Final Project
Cody Gilbert

This module implements the BERT model using the Python fast-bert module.

Selected code is modified from the fast-bert example code
by Kaushal Trivedi
located in 
https://github.com/kaushaltrivedi/fast-bert/blob/master/sample_notebooks/new-toxic-multilabel.ipynb
'''

import sys
import os
import pandas as pd
import numpy as np
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

from fast_bert.data_cls import BertDataBunch
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_PATH = Path('./data/')
DATA_PATH.mkdir(exist_ok=True)
LABEL_PATH = Path('./labels/')
LABEL_PATH.mkdir(exist_ok=True)
MODEL_PATH=Path('./models/')
MODEL_PATH.mkdir(exist_ok=True)
OUTPUT_DIR = Path('./output/')
OUTPUT_DIR.mkdir(exist_ok=True)

max_rows = 10000
vocab_size = 5000
featurize = False
max_review_length = 200
test_fraction = 0.25
valid_fraction = 0.25
embedding_vector_length = 32
seed = 42
training_file = str(DATA_PATH / 'train.csv')
validation_file = str(DATA_PATH / 'val.csv')
test_file = str(DATA_PATH / 'test.csv')
labels_file = str(LABEL_PATH / 'labels.csv')
raw_reviews_file = 'train-balanced-sarcasm.csv'
metrics = [{'name': 'accuracy', 'function': accuracy}]

label_map = { 0: 'neg',
              1: 'pos'}

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

# Create training files
logger.debug('Importing features')
features_df = pd.read_csv(raw_reviews_file, index_col=None, nrows=max_rows)
features_df = features_df[['label', 'comment']].rename(columns={'comment': 'text'})
features_df['label'] = features_df['label'].map(label_map)
features_df.index.name = 'index'
labels = pd.DataFrame(features_df['label'].unique())
train_set, test_set = train_test_split(features_df, test_size=test_fraction, random_state=seed)
train_set, valid_set = train_test_split(train_set, test_size=valid_fraction, random_state=seed)

train_set.to_csv(training_file, index=True)
valid_set.to_csv(validation_file, index=True)
test_set.to_csv(test_file, index=True)
labels.to_csv(labels_file, index=False, header=False)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
outputs = model(input_ids, labels=labels)
loss, logits = outputs[:2]

# Setup and train the BERT model
device_cuda = torch.device('cpu')
logger.debug('Create Data Bunch')
databunch = BertDataBunch(DATA_PATH, 
                          LABEL_PATH,
                          tokenizer='bert-base-uncased',
                          train_file='train.csv',
                          val_file='val.csv',
                          label_file='labels.csv',
                          text_col='text',
                          label_col='label',
                          max_seq_length=512,
                          batch_size_per_gpu=128,
                          multi_gpu=False,
                          multi_label=False,
                          model_type='bert')
logger.debug('Opening Leaner')
learner = BertLearner.from_pretrained_model(
                        databunch,
                        pretrained_path='bert-base-uncased',
                        metrics=metrics,
                        device=device_cuda,
                        logger=logger,
                        output_dir=OUTPUT_DIR,
                        finetuned_wgts_path=None,
                        warmup_steps=500,
                        multi_gpu=False,
                        is_fp16=False,
                        multi_label=False,
                        logging_steps=50)
logger.debug('Tuning Model')
learner.fit(epochs=6,
            lr=6e-5,
            validate=True,  # Evaluate the model after each epoch
            schedule_type="warmup_cosine",
            optimizer_type="lamb")
learner.save_model()
            