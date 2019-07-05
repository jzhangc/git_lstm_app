"""
Custom functions for the deep learning pratices
"""
# ------ libraries ------
import logging
import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout  # fully connnected layer
from sklearn.metrics import roc_curve, auc  # calculate ROC-AUC
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt  # to plot ROC-AUC
# import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR)  # disable tensorflow ERROR message


# ------ functions ------
def training_test_spliter(data, training_percent=0.8, random_state=None,
                          min_max_scaling=False, scale_column_as_y=None, scale_column_to_exclude=None, scale_range=(0, 1)):
    """
    Purpose:
        This funciton takes an pandas DataFrame, randomly resamples the data and
        split it into trianing and test data sets.
        The function includes a Min-Max scaling functionality, separately for Y and X.

    Return:
        Pandas DataFrame (for now) for training and test data.
        The X and Y scalers are also returned

    Arguments:
        data: input data
        training_percent: percentage of the full data to be the training
        random_state: seed for resampling RNG
        min_max_scaling: if to do a Min_Max scaling to the data
        scale_column_as_y: column(s) to use as outcome for scaling
        scale_column_to_exclude: has to be a list, the name of the columns 
                                to remove from the X columns for scaling. 
                                makes sure to also inlcude the y column(s)
        scale_range: the Min_Max range
    """
    scaler_X, scaler_Y = None, None

    # normalization if needed
    if min_max_scaling:
        selected_cols = scale_column_as_y + scale_column_to_exclude
        # NOTE: set removes duplicates but disturbes order
        # HINT: use dict if order matters
        selected_cols = list(set(selected_cols))
        if all(selected_col in data.columns for selected_col in selected_cols):
            scaler_X = MinMaxScaler(feature_range=scale_range)
            data[data.columns[~data.columns.isin(scale_column_to_exclude)]] = scaler_X.fit_transform(
                data[data.columns[~data.columns.isin(scale_column_to_exclude)]])

            if scale_column_as_y is not None:
                scaler_Y = MinMaxScaler(feature_range=scale_range)
                data[data.columns[data.columns.isin(scale_column_as_y)]] = scaler_Y.fit_transform(
                    data[data.columns[data.columns.isin(scale_column_as_y)]])
        else:
            print(
                'Not all columns are found in the input DataFrame. Proceed without normalization\n')

    # split the data
    training = data.sample(frac=training_percent, random_state=random_state)
    test = data.iloc[~data.index.isin(training.index), :]
    return training, test, scaler_X, scaler_Y
