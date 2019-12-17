# _*_ coding: utf-8 _*_

import os
import sys
import torch
from torch.nn import functional as F
import numpy as np
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
from torchtext.vocab import FastText
import pandas as pd
from utils import DataFrameDataset


def load_dataset(test_sen=None):

    # read and import csv file. Skip rows due to memory limit and some rows have bad formatting. TODO need to clean csv
    fields = ['label', 'comment']
    df = pd.read_csv('../../data/train-balanced-sarcasm_cleaned.csv', skipinitialspace=True, usecols=fields,
                     skiprows=range(50000, 1010827))
    # df.to_csv('./data/train_data.txt', index=False, sep=',', header=None)

    tokenize = lambda x: x.split()
    TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True,
                      fix_length=200)
    LABEL = data.LabelField()

    fields = {'label': LABEL, 'comment': TEXT}
    train_data = DataFrameDataset.DataFrameDataset(df, fields)  # convert df into dataset
    train_data, test_data = train_data.split()
    TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))
    LABEL.build_vocab(train_data)

    word_embeddings = TEXT.vocab.vectors
    print("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print("Label Length: " + str(len(LABEL.vocab)))

    train_data, valid_data = train_data.split()  # Further splitting of training_data to create new training_data & validation_data
    train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size=32,
                                                                   sort_key=lambda x: len(x.comment), repeat=False,
                                                                   shuffle=True)

    vocab_size = len(TEXT.vocab)

    return vocab_size, word_embeddings, train_iter, valid_iter, test_iter


if __name__ == '__main__':
    vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_dataset()

