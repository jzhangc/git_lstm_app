"""
Custom functions for the deep learning pratices
"""

# ------ libraries ------
import logging

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt  # to plot ROC-AUC
from sklearn.metrics import auc, roc_curve  # calculate ROC-AUC
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR)  # disable tensorflow ERROR message


# ------ functions ------
def training_test_spliter_final(data,
                                training_percent=0.8, random_state=None,
                                man_split=False, man_split_colname=None,
                                man_split_testset_value=None,
                                x_standardization=True,
                                x_scale_column_to_exclude=None,
                                y_min_max_scaling=False, y_column_to_scale=None,
                                y_scale_range=(0, 1)):
    """
    # Purpose:
        This is a final verion of the training_test_spliter. 
        This version splits the data into training and test prior to Min-Max scaling.
        The z score standardization is used for X standardization

    # Return:
        Pandas DataFrame (for now) for training and test data.
        Y scalers for training and test data sets are also returned.
        Order: training (np.array), test (np.array), training_scaler_X, training_scaler_Y

    # Arguments:
        data: Pnadas DataFrame. Input data.
        man_split: boolean. If to manually split the data into training/test sets.
        man_split_colname: string. Set only when fixed_split=True, the variable name for the column to use for manual splitting.
        man_split_testset_value: list. Set only when fixed_split=True, the splitting variable values for test set.
        training_percent: float. percentage of the full data to be the training
        random_state: int. seed for resampling RNG
        x_standardization: boolean. if to center scale (z score standardization) the input X data 
        x_scale_column_to_exclude: list. the name of the columns
                                to remove from the X columns for scaling.
                                makes sure to also inlcude the y column(s)
        y_column_to_scale: list. column(s) to use as outcome for scaling
        y_min_max_scaling: boolean. For regression study, if to do a Min_Max scaling to outcome
        y_scale_range: two-tuple. the Min_Max range.

    # Details:
        The data normalization is applied AFTER the training/test splitting
        The x_standardization is z score standardization ("center and scale"): (x - mean(x))/SD
        The y_min_max_scaling is min-max nomalization
        When x_standardization=True, the test data is standardized using training data mean and SD.
        When y_min_max_scaling=True, the test data is scaled using training data max-min parameters.

    # Examples
    1. with normalization
        training, test, training_scaler_X, training_scaler_Y = training_test_spliter_final(
            data=raw, random_state=1,
            man_split=True, man_split_colname='subject', man_split_testset_value=selected_features[0],
            x_standardization=True,
            x_scale_column_to_exclude=['subject', 'PCL', 'group'],
            y_min_max_scaling=True,
            y_column_to_scale=['PCL'], y_scale_range=(0, 1))

    2. without noralization
        training, test, _, _ = training_test_spliter_final(
            data=raw, random_state=1,
            man_split=True, man_split_colname='subject', man_split_testset_value=selected_features[1],
            x_standardization=False,
            y_min_max_scaling=False)

    """
    # argument check
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input needs to be a pandas DataFrame.")

    if x_standardization:
        if not isinstance(x_scale_column_to_exclude, list):
            raise ValueError(
                'x_scale_column_to_exclude needs to be a list.')
    if y_min_max_scaling:
        if not isinstance(y_column_to_scale, list):
            raise ValueError(
                'y_column_to_scale needs to be a list.')

    if man_split:
        if (not man_split_colname) or (not man_split_testset_value):
            raise ValueError(
                'set man_split_colname and man_split_testset_value when man_split=True.')
        else:
            if not isinstance(man_split_colname, str):
                raise ValueError('man_split_colname needs to be a string.')
            if not isinstance(man_split_testset_value, list):
                raise ValueError(
                    'man_split_colvaue needs to be a list.')
            if not all(test_value in list(data[man_split_colname]) for test_value in man_split_testset_value):
                raise ValueError(
                    'One or all man_split_test_value missing from the splitting variable.')

    # split
    if man_split:
        # .copy() to make it explicit that it is a copy, to avoid Pandas SettingWithCopyWarning
        training = data.loc[~data[man_split_colname].isin(
            man_split_testset_value), :].copy()
        test = data.loc[data[man_split_colname].isin(
            man_split_testset_value), :].copy()
    else:
        training = data.sample(frac=training_percent,
                               random_state=random_state)
        test = data.iloc[~data.index.isin(training.index), :].copy()

    # normalization if needed
    # set the variables
    training_scaler_X, training_scaler_Y, test_scaler_Y = None, None, None
    if x_standardization:
        if all(selected_col in data.columns for selected_col in x_scale_column_to_exclude):
            training_scaler_X = StandardScaler()
            training[training.columns[~training.columns.isin(x_scale_column_to_exclude)]] = training_scaler_X.fit_transform(
                training[training.columns[~training.columns.isin(x_scale_column_to_exclude)]])
            test[test.columns[~test.columns.isin(x_scale_column_to_exclude)]] = training_scaler_X.transform(
                test[test.columns[~test.columns.isin(x_scale_column_to_exclude)]])
        else:
            print(
                'Not all columns are found in the input X. Proceed without X standardization. \n')

    if y_min_max_scaling:
        if all(selected_col in data.columns for selected_col in y_column_to_scale):
            training_scaler_Y = MinMaxScaler(feature_range=y_scale_range)
            training[training.columns[training.columns.isin(y_column_to_scale)]] = training_scaler_Y.fit_transform(
                training[training.columns[training.columns.isin(y_column_to_scale)]])
            test[test.columns[test.columns.isin(y_column_to_scale)]] = training_scaler_Y.transform(
                test[test.columns[test.columns.isin(y_column_to_scale)]])
        else:
            print(
                'Y column to scale not found. Proceed without Y scaling. \n')

    return training, test, training_scaler_X, training_scaler_Y


def training_test_spliter(data,
                          training_percent=0.8, random_state=None,
                          man_split=False, man_split_colname=None,
                          man_split_testset_value=None,
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
        data: Pnadas DataFrame. Input data.
        man_split: boolean. If to manually split the data into training/test sets.
        man_split_colname: string. Set only when fixed_split=True, the variable name for the column to use for manual splitting.
        man_split_testset_value: list. Set only when fixed_split=True, the splitting variable values for test set.
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
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input needs to be a pandas DataFrame.")
    if not all(isinstance(scale_list, list) for scale_list in [scale_column_as_y, scale_column_to_exclude]):
        raise ValueError(
            'scale_column_as_y and scale_column_to_exclude need to be list.')
    if man_split:
        if (not man_split_colname) or (not man_split_testset_value):
            raise ValueError(
                'set man_split_colname and man_split_testset_value when man_split=True.')
        else:
            if not isinstance(man_split_colname, str):
                raise ValueError('man_split_colname needs to be a string.')
            if not isinstance(man_split_testset_value, list):
                raise ValueError(
                    'man_split_colvaue needs to be a list.')
            if not all(test_value in list(data[man_split_colname]) for test_value in man_split_testset_value):
                raise ValueError(
                    'One or all man_split_test_value missing from the splitting variable.')

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
    if man_split:
        training = data.loc[~data[man_split_colname].isin(
            man_split_testset_value), :]
        test = data.loc[data[man_split_colname].isin(
            man_split_testset_value), :]
    else:
        training = data.sample(frac=training_percent,
                               random_state=random_state)
        test = data.iloc[~data.index.isin(training.index), :]

    return training, test, scaler_X, scaler_Y


def inverse_norm_y(training_y, test_y, scaler):
    """
    (to be deprecated)
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


def training_test_spliter_new(data,
                              training_percent=0.8, random_state=None,
                              man_split=False, man_split_colname=None,
                              man_split_testset_value=None,
                              min_max_scaling=False, scale_column_as_y=None,
                              scale_column_to_exclude=None, scale_range=(0, 1)):
    """
    # Purpose:
        This is a new verion of the training_test_spliter. This version splits the data into
        training and test prior to Min-Max scaling.

    # Return:
        Pandas DataFrame (for now) for training and test data.
        Y scalers for training and test data sets are also returned. 
        Order: training (np.array), test (np.array), training_scaler_Y, test_scaler_Y 

    # Arguments:
        data: Pnadas DataFrame. Input data.
        man_split: boolean. If to manually split the data into training/test sets.
        man_split_colname: string. Set only when fixed_split=True, the variable name for the column to use for manual splitting.
        man_split_testset_value: list. Set only when fixed_split=True, the splitting variable values for test set.
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
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input needs to be a pandas DataFrame.")
    if not all(isinstance(scale_list, list) for scale_list in [scale_column_as_y, scale_column_to_exclude]):
        raise ValueError(
            'scale_column_as_y and scale_column_to_exclude need to be list.')
    if man_split:
        if (not man_split_colname) or (not man_split_testset_value):
            raise ValueError(
                'set man_split_colname and man_split_testset_value when man_split=True.')
        else:
            if not isinstance(man_split_colname, str):
                raise ValueError('man_split_colname needs to be a string.')
            if not isinstance(man_split_testset_value, list):
                raise ValueError(
                    'man_split_colvaue needs to be a list.')
            if not all(test_value in list(data[man_split_colname]) for test_value in man_split_testset_value):
                raise ValueError(
                    'One or all man_split_test_value missing from the splitting variable.')

    # split
    if man_split:
        # .copy() to make it explicit that it is a copy, to avoid Pandas SettingWithCopyWarning
        training = data.loc[~data[man_split_colname].isin(
            man_split_testset_value), :].copy()
        test = data.loc[data[man_split_colname].isin(
            man_split_testset_value), :].copy()
    else:
        training = data.sample(frac=training_percent,
                               random_state=random_state)
        test = data.iloc[~data.index.isin(training.index), :].copy()

    # normalization if needed
    # set the variables
    scaler_X, training_scaler_Y, test_scaler_Y = None, None, None

    if min_max_scaling:
        selected_cols = scale_column_as_y + scale_column_to_exclude
        if all(selected_col in data.columns for selected_col in selected_cols):
            scaler_X = MinMaxScaler(feature_range=scale_range)
            training[training.columns[~training.columns.isin(scale_column_to_exclude)]] = scaler_X.fit_transform(
                training[training.columns[~training.columns.isin(scale_column_to_exclude)]])
            test[test.columns[~test.columns.isin(scale_column_to_exclude)]] = scaler_X.fit_transform(
                test[test.columns[~test.columns.isin(scale_column_to_exclude)]])

            if scale_column_as_y is not None:
                training_scaler_Y = MinMaxScaler(feature_range=scale_range)
                training[training.columns[training.columns.isin(scale_column_as_y)]] = training_scaler_Y.fit_transform(
                    training[training.columns[training.columns.isin(scale_column_as_y)]])

                test_scaler_Y = MinMaxScaler(feature_range=scale_range)
                test[test.columns[test.columns.isin(scale_column_as_y)]] = test_scaler_Y.fit_transform(
                    test[test.columns[test.columns.isin(scale_column_as_y)]])
        else:
            print(
                'Not all columns are found in the input DataFrame. Proceed without normalization\n')

    return training, test, training_scaler_Y, test_scaler_Y
