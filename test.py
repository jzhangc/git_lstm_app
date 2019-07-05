"""
to test small things
"""
import math
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
# StratifiedKFold should be used for classification problems
# StratifiedKFold makes sure the fold has an equal representation of the classes
from sklearn.model_selection import KFold

from custom_functions.cv_functions import (PdDataFrameTypeError, NpArrayShapeError, idx_func,
                                           longitudinal_cv_xy_array,
                                           lstm_train_eval)
from custom_functions.data_processing import training_test_spliter
from custom_functions.util_functions import logging_func


# ------ test functions ------
def lstm_cv(input, Y_colnames, remove_colnames, n_features,
            cv_n_folds=10,  cv_random_state=None,
            lstm_mode="simple"):
    """
    Purpose:
        This is the main function for k-fold cross validation for LSTM RNN

    Arguments:
        input
        Y_colnames
        remove_colnames
        cv_n_folds
        cv_random_state
        lstm_mode

    Return
    An ensemble of LSTM RNN models that can be used for ensemble prediction
    """
    # arugment checks
    if not isinstance(input, pd.DataFrame):
        raise PdDataFrameTypeError("Inoput needs to be a pandas DataFrame.")
    if not isinstance(Y_colnames, list) or not isinstance(remove_colnames, list):
        raise TypeError("Y_colnames and remove_colnames need to be list type.")

    # set up the x y array data
    X, Y = longitudinal_cv_xy_array(input=input, Y_colnames=Y_colnames,
                                    remove_colnames=remove_colnames, n_features=n_features)
    return None


def lstm_member_eval(models, n_numbers, testX, testY):
    """
    # Purpose:
        The function evaluates a subset of models from the CV model ensemble

    # Arguments:
        models: list. CV model ensemble
        n_numbers: int. The first n number of models
    """

    return None


def lstm_ensemble_predict(models, testX):
    """
    # Purpose:
        Make predictions using an ensemble of lstm models.

    # Arguments:
        models: a list of lstm models
        testX: test X. Needs to be a numpy ndarray object.
    """
    # argument check
    if not isinstance(testX, np.array):
        raise TypeError("testX needs to be a numpy array.")
    if not len(testX.shape) == 3:
        raise NpArrayShapeError("testX needs to be in 3D shape.")

    # testX
    yhats = [m.predict(testX) for m in models]
    yhats = np.array(yhats)

    # argmax the results
    result = np.argmax(yhats)

    return result


# ------ script ------
# ---- working directory
main_dir = os.path.abspath('./')
dat_dir = os.path.join(main_dir, 'data')
res_dir = os.path.join(main_dir, 'results')
log_dir = os.path.join(main_dir, 'log')

# ---- setup logger
logger = logging_func(filepath=os.path.join(
    log_dir, 'test.log'))

# ---- import data
raw = pd.read_csv(os.path.join(
    dat_dir, 'lstm_aec_phases_freq1.csv'), engine='python')
raw.iloc[0:5, 0:5]

# ---- generate training and test sets with min-max normalization
training, test, scaler_X, scaler_Y = training_test_spliter(
    data=raw, training_percent=0.9, random_state=10, min_max_scaling=True,
    scale_column_as_y=['PCL'],
    scale_column_to_exclude=['subject', 'PCL', 'group'])

# ---- test k-fold data sampling
X, Y, cv_train_idx, cv_test_idx = idx_func(input=training, n_features=8, Y_colnames=['PCL'],
                                           remove_colnames=['subject', 'group'], n_folds=10, random_state=5)

X.shape  # n_samples x n_timepoints x n_features: 29x2x8
Y.shape  # 29x1
len(cv_train_idx[0])

cv_train_X, cv_train_Y = X[cv_train_idx[0]], Y[cv_train_idx[0]]
cv_test_X, cv_test_Y = X[cv_test_idx[0]], Y[cv_test_idx[0]]

# modelling
cv_m, cv_m_history, cv_test_rmse = lstm_train_eval(trainX=cv_train_X, trainY=cv_train_Y,
                                                   testX=cv_test_X, testY=cv_test_Y, hidden_units=6, epochs=400, batch_size=29,
                                                   plot=True, filepath=os.path.join(res_dir, 'cv_simple_loss.pdf'),
                                                   plot_title='Simple LSTM model',
                                                   ylabel='MSE',
                                                   verbose=False)
