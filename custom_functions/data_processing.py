"""
Custom functions for the deep learning pratices
"""

# ------ libraries ------
import logging

import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout  # fully connnected layer
from keras.models import Sequential, load_model
from matplotlib import pyplot as plt  # to plot ROC-AUC
from sklearn.metrics import auc, roc_curve  # calculate ROC-AUC
from sklearn.preprocessing import MinMaxScaler

# import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR)  # disable tensorflow ERROR message


# ------ functions ------
def training_test_spliter(data, training_percent=0.8, random_state=None,
                          min_max_scaling=False, scale_column_as_y=None,
                          scale_column_to_exclude=None, scale_range=(0, 1)):
    """
    # Purpose:
        This funciton takes an pandas DataFrame, randomly resamples the data and
        split it into trianing and test data sets.
        The function includes a Min-Max scaling functionality, separately for Y and X.

    # Return:
        Pandas DataFrame (for now) for training and test data.
        The X and Y scalers are also returned

    # Arguments:
        data: . Pnadas DataFrame. input data.
        training_percent: float. percentage of the full data to be the training
        random_state: int. seed for resampling RNG
        min_max_scaling: boolean. if to do a Min_Max scaling to the data
        scale_column_as_y: list. column(s) to use as outcome for scaling
        scale_column_to_exclude: list. the name of the columns 
                                to remove from the X columns for scaling. 
                                makes sure to also inlcude the y column(s)
        scale_range: two-tuple. the Min_Max range.
    """
    # argument check
    if not isinstance(input, pd.DataFrame):
        raise TypeError("Inoput needs to be a pandas DataFrame.")
    if not all(isinstance(scale_list, pd.DataFrame) for scale_list in [scale_column_as_y, scale_column_to_exclude]):
        raise ValueError(
            'scale_column_as_y and scale_column_to_exclude need to be list.')

    # set the variables
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


def inverse_norm_y(training_y, test_y, scaler):
    """
    # Purpose:
        This function inverses the min-max normalized y values

    # Return:
        Training and test data inverse-transformed, in numpy array format

    # Details:
        The length of the input scaler is the combination of the input data, 
        e.g. training_y and test_y

    # Arguments:
        training_y: input training y data. 
        test_y: input test y data
        scaler: the scaler that used for the min-max normalization, produced long with the data through
                function training_test_spliter()

    NOTE: if not already , the data will be converted to a numpy array
    """
    dat = np.concatenate([np.array(training_y), np.array(test_y)])
    dat = scaler.inverse_transform(dat.reshape(dat.shape[0], 1))
    dat = dat.reshape(dat.shape[0], )
    training_y_out = dat[0:training_y.shape[0]]
    test_y_out = dat[training_y.shape[0]:]
    return training_y_out, test_y_out
