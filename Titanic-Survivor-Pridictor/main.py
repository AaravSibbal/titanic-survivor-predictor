
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow as tf

import tensorflow.compat.v2.feature_column as fc

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
#print(dftrain.head())
y_train = dftrain.pop("survived")
y_eval = dfeval.pop("survived")

CATEGORICAL_COLUMN = ["sex", "n_siblings_spouses", "parch", "class", "deck", "embark_town", "alone"]
NUMERIC_COLUMN = ["age", "fare"]

feature_columns = []
for feature_name in CATEGORICAL_COLUMN: 
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
for feature_name in NUMERIC_COLUMN:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

#print(feature_columns)

def make_input_fn(data_df, label_df, num_epochs=100, shuffle=True, batch_size=32):
    def input_function(): #inner function, this will be returned 
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000) #randomize the order of data
        ds = ds.batch(batch_size).repeat(num_epochs) #splits dataset into batches of 32 and repeat process for number of epochs
        return ds # returns a batch of the dataset
    return input_function #returns a function object for use
train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

linear_est.train(train_input_fn) #train
result = linear_est.evaluate(eval_input_fn)

clear_output()
print(result['accuracy'])

    