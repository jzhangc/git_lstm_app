"""
to test small things
"""
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import math
import os
import datetime

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import History  # for input argument type check
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
# StratifiedKFold should be used for classification problems
# StratifiedKFold makes sure the fold has an equal representation of the classes
from sklearn.model_selection import KFold
# from sklearn.preprocessing import MinMaxScaler, StandardScaler

from custom_functions.cv_functions import (idx_func, longitudinal_cv_xy_array,
                                           lstm_cv_train, lstm_ensemble_eval,
                                           lstm_ensemble_predict)
from custom_functions.data_processing import training_test_spliter_final
from custom_functions.plot_functions import y_yhat_plot
from custom_functions.util_functions import logging_func

# ------ test functions ------
# NOTE: this should be a custom class

# ------ data processing ------
# ---- working directory
main_dir = os.path.abspath('./')
dat_dir = os.path.join(main_dir, 'data')
log_dir = os.path.join(main_dir, 'log')

# ---- setup logger
logger = logging_func(filepath=os.path.join(log_dir, 'test.log'))

# ---- import data
freq = 'freq6'
raw = pd.read_csv(os.path.join(
    dat_dir, 'new_lstm_aec_phase1_base_'+freq+'.csv'), engine='python')

raw.iloc[0:5, 0:5]
raw.shape

# ---- key variable
n_features = 10
n_folds = 10
selected_features = [["PP12", "PN05", "PN11"],
                     ["PP19", "PP18", "PN23"],
                     ["PP15", "PN16", "PN18"],
                     ["PP15", "PP24", "PN11"],
                     ["PN17", "PN19", "PP13"],
                     ["PP05", "PN03", "PP20"],
                     ["PP20", "PP21", "PN11"]]  # order: freq 1~7

# result directory
res_dir = os.path.join(main_dir, 'results/new_lstm_phase1_base', freq)


# ---- generate training and test sets with min-max normalization
# new spliting
training, test, _, _ = training_test_spliter_final(
    data=raw, random_state=1,
    man_split=True, man_split_colname='subject', man_split_testset_value=selected_features[5],
    x_standardization=False,
    y_min_max_scaling=False)

# below: as an example for converting the normalized Y back to measured values
# _, testY = inverse_norm_y(training_y=trainingY, test_y=testY, scaler=scaler_Y)
# ---- test k-fold data sampling
cv_train_idx, cv_test_idx = idx_func(input=training, n_features=n_features, Y_colnames=['PCL'],
                                     remove_colnames=['subject', 'group'], n_folds=n_folds, random_state=800)

len(cv_train_idx[0])  # 26

# ------ cross-validation modelling ------
# --- modelling ---
cv_m_ensemble, cv_m_history_ensemble, cv_pred_ensemble, cv_test_rmse_ensemble, cv_test_rsq_ensemble = list(
), list(), list(), list(), list()
for i in range(n_folds):
    """
    This CV training loop standardizes X and standardizes Y every iteration for each CV fold.
    """
    fold_id = str(i+1)
    print('fold: ', fold_id)
    cv_train, cv_test = training.iloc[cv_train_idx[i],
                                      :].copy(), training.iloc[cv_test_idx[i], :].copy()

    # below: X standardization
    cv_train_scaler_X = StandardScaler()
    cv_train[cv_train.columns[~cv_train.columns.isin(['subject', 'PCL', 'group'])]] = cv_train_scaler_X.fit_transform(
        cv_train[cv_train.columns[~cv_train.columns.isin(['subject', 'PCL', 'group'])]])
    cv_test[cv_test.columns[~cv_test.columns.isin(['subject', 'PCL', 'group'])]] = cv_train_scaler_X.transform(
        cv_test[test.columns[~cv_test.columns.isin(['subject', 'PCL', 'group'])]])

    # below: Y min-max scaling
    cv_train_scaler_Y = MinMaxScaler(feature_range=(0, 1))
    cv_train[cv_train.columns[cv_train.columns.isin(['PCL'])]] = cv_train_scaler_Y.fit_transform(
        cv_train[cv_train.columns[cv_train.columns.isin(['PCL'])]])
    cv_test[cv_test.columns[cv_test.columns.isin(['PCL'])]] = cv_train_scaler_Y.fit_transform(
        cv_test[cv_test.columns[cv_test.columns.isin(['PCL'])]])

    # transform into numpy arrays
    cv_train_X, cv_train_Y = longitudinal_cv_xy_array(input=cv_train, Y_colnames=['PCL'],
                                                      remove_colnames=['subject', 'group'], n_features=n_features)
    cv_test_X, cv_test_Y = longitudinal_cv_xy_array(input=cv_test, Y_colnames=['PCL'],
                                                    remove_colnames=['subject', 'group'], n_features=n_features)

    # train
    cv_m, cv_m_history, cv_pred, cv_m_test_rmse, cv_m_test_rsq = lstm_cv_train(trainX=cv_train_X, trainY=cv_train_Y,
                                                                               testX=cv_test_X, testY=cv_test_Y,
                                                                               lstm_model='stacked',
                                                                               study_type='n_to_one',
                                                                               prediction_inverse=True, y_scaler=cv_train_scaler_Y,
                                                                               hidden_units=25, epochs=150, batch_size=29,
                                                                               output_activation='sigmoid',
                                                                               log_dir=os.path.join(
                                                                                   res_dir, 'fit', 'cv_fold_'+fold_id),
                                                                               plot=True,
                                                                               filepath=os.path.join(
                                                                                   res_dir, 'cv_simple_loss_fold_'+fold_id+'.pdf'),
                                                                               plot_title=None,
                                                                               xlabel=None,
                                                                               ylabel=None,
                                                                               verbose=False)
    cv_m.save(os.path.join(res_dir, 'cv_models',
                           freq+'_cv_'+'fold_'+str(i+1)+'.h5'))
    cv_m_ensemble.append(cv_m)
    cv_m_history_ensemble.append(cv_m_history)
    cv_pred_ensemble.append(cv_pred)
    cv_test_rmse_ensemble.append(cv_m_test_rmse)
    cv_test_rsq_ensemble.append(cv_m_test_rsq)


# --- CV evaluation ---
cv_rmse_mean = np.mean(cv_test_rmse_ensemble)
cv_rmse_sd = np.std(cv_test_rmse_ensemble)
cv_rmse_sem = cv_rmse_sd/math.sqrt(10)
cv_pred_ensemble
cv_test_rmse_ensemble
top5_index = list((np.argsort(cv_test_rmse_ensemble)[:5]))
cv_rmse_mean
cv_rmse_sd
cv_rmse_sem

# ------ final modelling and evaluation ------
# data transformation
training_scaler_X = StandardScaler()
x_scale_column_to_exclude = ['subject', 'PCL', 'group']
training[training.columns[~training.columns.isin(x_scale_column_to_exclude)]] = training_scaler_X.fit_transform(
    training[training.columns[~training.columns.isin(x_scale_column_to_exclude)]])
test[test.columns[~test.columns.isin(x_scale_column_to_exclude)]] = training_scaler_X.transform(
    test[test.columns[~test.columns.isin(x_scale_column_to_exclude)]])

training_scaler_Y = MinMaxScaler(feature_range=(0, 1))
y_column_to_scale = ['PCL']
training[training.columns[training.columns.isin(y_column_to_scale)]] = training_scaler_Y.fit_transform(
    training[training.columns[training.columns.isin(y_column_to_scale)]])
test[test.columns[test.columns.isin(y_column_to_scale)]] = training_scaler_Y.transform(
    test[test.columns[test.columns.isin(y_column_to_scale)]])

# turn data into numpy arrays
trainingX, trainingY = longitudinal_cv_xy_array(input=training, Y_colnames=['PCL'],
                                                remove_colnames=['subject', 'group'], n_features=n_features)
testX, testY = longitudinal_cv_xy_array(input=test, Y_colnames=['PCL'],
                                        remove_colnames=['subject', 'group'], n_features=n_features)

trainingX.shape  # n_samples x n_timepoints x n_features
trainingY.shape  # 29x1
training_scaler_Y.inverse_transform(testY)

# modelling
final_m, final_m_history, final_pred, final_m_test_rmse, final_m_test_rsq = lstm_cv_train(trainX=trainingX, trainY=trainingY,
                                                                                          testX=testX, testY=testY,
                                                                                          lstm_model='stacked',
                                                                                          study_type='n_to_one',
                                                                                          prediction_inverse=True, y_scaler=training_scaler_Y,
                                                                                          hidden_units=50, epochs=150, batch_size=29,
                                                                                          output_activation='sigmoid',
                                                                                          log_dir=os.path.join(
                                                                                              res_dir, 'fit', 'final_model'),
                                                                                          plot=True,
                                                                                          filepath=os.path.join(
                                                                                              res_dir, 'final_model.pdf'),
                                                                                          plot_title=None,
                                                                                          xlabel=None,
                                                                                          ylabel=None,
                                                                                          verbose=True)


final_pred
final_m_test_rmse

# ------ true test realm ------
