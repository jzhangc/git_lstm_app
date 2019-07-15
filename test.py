"""
to test small things
"""
import math
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import mean_squared_error, accuracy_score
# StratifiedKFold should be used for classification problems
# StratifiedKFold makes sure the fold has an equal representation of the classes
from sklearn.model_selection import KFold
from keras.utils import to_categorical

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


def lstm_member_eval(models, n_numbers, testX, testY, outcome_type='regression'):
    """
    # Purpose:
        The function evaluates a subset of models from the CV model ensemble

    # Arguments:
        models: list. CV model ensemble
        n_numbers: int. The first n number of models
        outcome_type: string. The outcome type of the study, 'regression' or 'classification'
    """
    # subsetting model ensemble
    subset = models[:n_numbers]

    # prediction
    yhats = lstm_ensemble_predict(models=subset, testX=testX)

    # calculate acc or rmse
    if outcome_type == 'regression':
        res = math.sqrt(mean_squared_error(y_true=testY, y_pred=yhats))
    else:
        res = accuracy_score(y_true=testY, y_pred=yhats)

    return res


def lstm_ensemble_predict(models, testX, outcome_type='regression'):
    """
    # Purpose:
        Make predictions using an ensemble of lstm models.

    # Arguments:
        models: list. a list of lstm models
        testX: np.ndarray. test X. Needs to be a numpy ndarray object.
        outcome_type: string. the outcome type of the study, 'regression' or 'classification'

    # Reture:
        Indices to the max value of the sum of yhats.

    # Details:
        The function uses the models from the LSTM model ensemble (from k-fold CV process)
        to predict using input data X.

        For model_type='classification', instead of returning the prediction from each model used, 
        the function calculates the sum of yhat and returns the indices of the max sum value using
        np.argmax function. 

        Regarding axis values used by the numpy indexing for the some functions using 'axis=' argument,
        'axis=0' means "along row, or by column", and 'axis=1' means "along column, or by row".
    """
    # argument check
    if not isinstance(testX, np.ndarray):
        raise TypeError("testX needs to be a numpy array.")
    if not len(testX.shape) == 3:
        raise NpArrayShapeError("testX needs to be in 3D shape.")

    # testX
    yhats = [m.predict(testX) for m in models]
    yhats = np.array(yhats)

    if outcome_type == 'regression':
        result = yhats
    else:
        # sum
        sum = np.sum(yhats, axis=0)
        # argmax the results
        result = np.argmax(sum, axis=1)

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
n_folds = 10
X, Y, cv_train_idx, cv_test_idx = idx_func(input=training, n_features=8, Y_colnames=['PCL'],
                                           remove_colnames=['subject', 'group'], n_folds=n_folds, random_state=5)

X.shape  # n_samples x n_timepoints x n_features: 29x2x8
Y.shape  # 29x1
len(cv_train_idx[0])

cv_train_X, cv_train_Y = X[cv_train_idx[0]], Y[cv_train_idx[0]]
cv_test_X, cv_test_Y = X[cv_test_idx[0]], Y[cv_test_idx[0]]

# test
cv_m_ensemble, cv_m_history_ensemble, cv_m_test_rmse_ensemble = list(), list(), list()
for i in range(n_folds):
    fold_id = str(i+1)
    print('fold: ', fold_id, '\n')
    cv_train_X, cv_train_Y = X[cv_train_idx[i]], Y[cv_train_idx[i]]
    cv_test_X, cv_test_Y = X[cv_test_idx[i]], Y[cv_test_idx[i]]
    cv_m, cv_m_history, cv_m_test_rmse = lstm_train_eval(trainX=cv_train_X, trainY=cv_train_Y,
                                                         testX=cv_test_X, testY=cv_test_Y,
                                                         lstm_model='simple',
                                                         hidden_units=6, epochs=400, batch_size=29,
                                                         plot=False, filepath=os.path.join(res_dir, 'cv_simple_loss_fold_'+fold_id+'.pdf'),
                                                         plot_title='Simple LSTM model',
                                                         ylabel='MSE',
                                                         verbose=False)
    cv_m_ensemble.append(cv_m)
    cv_m_history_ensemble.append(cv_m_history)
    cv_m_test_rmse_ensemble.append(cv_m_test_rmse)


np.std(cv_m_test_rmse_ensemble)
np.mean(cv_m_test_rmse_ensemble)


tst = lstm_ensemble_predict(models=cv_m_ensemble, testX=cv_test_X)
math.sqrt(mean_squared_error(y_true=cv_test_Y, y_pred=tst))


models = cv_m_ensemble
testX = cv_test_X

cv_test_Y.shape
cv_test_X.shape

cv_test_X[2, :, :]

# testX
yhats = [m.predict(testX) for m in models]
yhats = np.array(yhats)

yhats.shape  # 10, 3, 1

# sum
sum = np.sum(yhats, axis=0)
yhats[0, :, :]


# argmax the results
sum.shape
sum[0, :]
result = np.argmax(sum, axis=1)  # how does this work??
sum[result]


dataX, datay = make_blobs(n_samples=55000, centers=3,
                          n_features=2, cluster_std=2, random_state=2)
X, newX = dataX[:5000, :], dataX[5000:, :]
y, newy = datay[:5000], datay[5000:]


dataX.shape
datay.shape

testy = to_categorical(y)
testy.shape
y.shape
