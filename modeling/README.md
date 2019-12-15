# Modeling Python Modules

This directory contains the Python Ver. 3 code used to execute the modeling software described by the final report.

## Python Modules

1. `bert_model.py`: Implements the BERT model using the Python fast-bert module.
2. `modeling.py`: Iterates over the Keras models defined in text_models.py and iterates over all considered model hyperparameters to find the model with the highest accuracy metrics.
3. `naive_bayes.py`: Implements Naive Bayes using the NLTK package
4. `text_models.py`: Defines a series of Keras models used by the modeling.py module for iterating over a grid of modeling parameters.

## Dependencies

These are the main third-party Python packages used for modeling:

1. [keras](https://keras.io/)
2. [sklearn](https://scikit-learn.org/stable/)
3. [torch](https://pytorch.org/)
4. [fast_bert](https://github.com/kaushaltrivedi/fast-bert): implements BERT for sequence classification

## Input Data

The modeling modules require two external plain-text data files that are not included here due to the size of the datasets:

1. `train-balanced-sarcasm.csv`: CSV corpus of labeled sarcasm comments [Source](https://www.kaggle.com/danofer/sarcasm)
1. `obscene_corpus.txt`: corpus of obscene/profane language [Source](https://www.cs.cmu.edu/~biglou/resources/bad-words.txt)

## Execution

Once the input data corpera have been placed in the module root directory, each Python module implementing models can be executed separately with

```python
python3 [module]
```

## Output Data

Each module will print results output to the screen as well as to the the `modeling.log` log file. The `modeling.py` will output results for each modeling iteration to a `results.csv` file containing the model validation metrics and hyperparameter state.