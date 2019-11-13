# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 16:12:40 2019

@author: Cody Gilbert
"""

import pandas as pd
import nltk, re, pprint
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import fasttext

# Set the stop word corpus
stop_words = set(stopwords.words('english'))

# Set the lemmatizer
lemmatizer = WordNetLemmatizer()

# Import the data
sarcasm_file = r"C:\Users\Cody Gilbert\Desktop\NLP\project\train-balanced-sarcasm.csv"
full_text_file = r"C:\Users\Cody Gilbert\Desktop\NLP\project\cleaned_text.txt"
clean_data_file = r"C:\Users\Cody Gilbert\Desktop\NLP\project\train-balanced-sarcasm_cleaned.csv"
model_file = r"C:\Users\Cody Gilbert\Desktop\NLP\project\embed_model.bin"
sarc_data_reader = pd.read_csv(sarcasm_file, index_col=False, chunksize=5000)

# extract special characters
def spec_char_adder(row):
    for col in ['comment', 'parent_comment']:
        text = row[col]
        if type(text) is type(''):
            # Note special chars
            if '?' in text:
                row['?_char_' + col] = True
            else:
                row['?_char_' + col] = False
            if '!' in text:
                row['!_char_' + col] = True
            else:
                row['!_char_' + col] = False
            row['upper_count_' + col] = 0
            clean_text = []
            for c in text:
                # Capture the number of uppercase words
                if c.isupper():
                    row['upper_count_' + col]  += 1
                if c.isalpha() or c == ' ' or c == '.':
                    clean_text.append(c.lower())
            row['cleaned_' + col] = ''.join(clean_text)
    return row

# Tokenize, Remove Stop words, and Lemmatize
def clean_words(row):
    for col in ['cleaned_comment', 'cleaned_parent_comment']:
        text = row[col]
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
                    clean_sentence = ' '.join(clean_sentence)
                    clean_text.append(clean_sentence)
            row[col] = '.'.join(clean_text)
    return(row)

#with open(full_text_file, 'w') as out_file:
#    clean_func = lambda x: clean_words(x, out_file)
#    for i, chunk in enumerate(sarc_data_reader):
#        print(f'Processing Chunk: {i}')
#        chunk = chunk.apply(spec_char_adder, axis=1)
#        chunk = chunk.apply(clean_words, axis=1)
#        for index, text in chunk['cleaned_comment'].items():
#            if type(text) is not type(0.2):
#                for sentence in text.split('.'):
#                    if sentence != '':
#                        out_file.write(sentence + '.\n')
#        for index, text in chunk['cleaned_parent_comment'].items():
#            if type(text) is not type(0.2):
#                for sentence in text.split('.'):
#                    if sentence != '':
#                        out_file.write(sentence + '.\n')
#        if i == 0:
#            chunk.to_csv(clean_data_file, index=False, header=True)
#        else:
#            chunk.to_csv(clean_data_file, mode='a', index=False, header=False)

# Train the model
model = fasttext.train_unsupervised(full_text_file)
model.save_model(model_file)






