"""
to test small things
"""
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from custom_functions.custom_exceptions import (NpArrayShapeError,
                                                PdDataFrameTypeError)
from custom_functions.lstm_functions import (bidirectional_lstm_m,
                                             simple_lstm_m, stacked_lstm_m)
from custom_functions.plot_functions import epochs_plot

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
raw = pd.read_csv(os.path.join(
    dat_dir, 'new_lstm_aec_phase1_base_freq1.csv'), engine='python')

raw.iloc[0:5, 0:5]

# ---- key variable
n_features = 7
n_folds = 10
freq = 'freq1'
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
training, test, training_scaler_X, training_scaler_Y, test_scaler_Y = training_test_spliter_final(
    data=raw, random_state=1,
    man_split=True, man_split_colname='subject', man_split_testset_value=selected_features[0],
    x_standardization=True,
    x_scale_column_to_exclude=['subject', 'PCL', 'group'],
    y_min_max_scaling=True,
    y_column_to_scale=['PCL'], y_scale_range=(0, 1))


# procesing
trainingX, trainingY = longitudinal_cv_xy_array(input=training, Y_colnames=['PCL'],
                                                remove_colnames=['subject', 'group'], n_features=n_features)
testX, testY = longitudinal_cv_xy_array(input=test, Y_colnames=['PCL'],
                                        remove_colnames=['subject', 'group'], n_features=n_features)


training_scaler_Y.inverse_transform(testY)

# below: as an example for converting the normalized Y back to measured values
# _, testY = inverse_norm_y(training_y=trainingY, test_y=testY, scaler=scaler_Y)
# ---- test k-fold data sampling
cv_train_idx, cv_test_idx = idx_func(input=training, n_features=n_features, Y_colnames=['PCL'],
                                     remove_colnames=['subject', 'group'], n_folds=n_folds, random_state=1)

trainingX.shape  # n_samples x n_timepoints x n_features
trainingY.shape  # 29x1
len(cv_train_idx[0])  # 26

# ------ cross-validation modelling ------
cv_m_ensemble, cv_m_history_ensemble, cv_m_test_rmse_ensemble, cv_m_test_rsq_ensemble = list(
), list(), list(), list()
for i in range(n_folds):
    fold_id = str(i+1)
    print('fold: ', fold_id)
    cv_train_X, cv_train_Y = trainingX[cv_train_idx[i]
                                       ], trainingY[cv_train_idx[i]]
    cv_test_X, cv_test_Y = trainingX[cv_test_idx[i]], trainingY[cv_test_idx[i]]
    cv_m, cv_m_history, cv_m_test_rmse, cv_m_test_rsq = lstm_cv_train(trainX=cv_train_X, trainY=cv_train_Y,
                                                                      testX=cv_test_X, testY=cv_test_Y,
                                                                      lstm_model='stacked',
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
    cv_m_ensemble.append(cv_m)
    cv_m_history_ensemble.append(cv_m_history)
    cv_m_test_rmse_ensemble.append(cv_m_test_rmse)
    cv_m_test_rsq_ensemble.append(cv_m_test_rsq)

cv_rmse_mean = np.mean(cv_m_test_rmse_ensemble)
cv_rmse_sem = np.std(cv_m_test_rmse_ensemble)/math.sqrt(n_folds)
cv_m_test_rmse_ensemble
cv_rmse_mean
cv_rmse_sem

cv_rsq_mean = np.mean(cv_m_test_rsq_ensemble)
cv_rsq_sem = np.std(cv_m_test_rsq_ensemble)/math.sqrt(n_folds)
cv_m_test_rsq_ensemble
cv_rsq_mean
cv_rsq_sem

# ------ prediction ------
# also features: inverse the predictions from normalized values to PCLs
# prediction for the test subjests
# yhats_testX = lstm_ensemble_predict(testX=testX,
#                                     models=cv_m_ensemble, model_index=[2, 5, 6, 8])
yhats_test = lstm_ensemble_predict(testX=testX, models=cv_m_ensemble)
# below: anix=0 means "along the row, by column"
# NOTE: below: real world utility only uses training scalers
yhats_test_conv = [training_scaler_Y.inverse_transform(
    ary) for ary in yhats_test]
yhats_test_conv = np.array(yhats_test_conv)
yhats_test_mean_conv = np.mean(yhats_test_conv, axis=0)
yhats_test_std = np.std(yhats_test_conv, axis=0)
yhats_test_sem = yhats_test_std/math.sqrt(n_folds)

# prediction for 80% test subjests
# yhats_trainingX = lstm_ensemble_predict(
#     models=cv_m_ensemble, testX=trainingX, model_index=[2, 5, 6, 8])
yhats_training = lstm_ensemble_predict(
    models=cv_m_ensemble, testX=trainingX)
yhats_training_conv = [training_scaler_Y.inverse_transform(
    ary) for ary in yhats_training]
yhats_training_conv = np.array(yhats_training_conv)
yhats_training_mean_conv = np.mean(yhats_training_conv, axis=0)
yhats_training_std = np.std(yhats_training_conv, axis=0)
yhats_training_sem = yhats_training_std/math.sqrt(n_folds)

# ------ eval ------
# below: calcuate RMSE and R2 (final, i.e. on the test data) using inversed y and yhat
# this RMSE is the score to report
testY_conv = training_scaler_Y.inverse_transform(testY)
rmse_yhats, rsq_yhats = list(), list()
for yhat_test in yhats_test_conv:  # NOTE: below: real world utility only uses training scalers
    rmse_yhat = math.sqrt(mean_squared_error(
        y_true=testY_conv, y_pred=yhat_test))
    rsq_yhat = r2_score(y_true=testY_conv, y_pred=yhat_test)
    rmse_yhats.append(rmse_yhat)
    rsq_yhats.append(rsq_yhat)


rmse_yhats = np.array(rmse_yhats)
rmse_yhats_mean = np.mean(rmse_yhats)
rmse_yhats_sem = np.std(rmse_yhats)/math.sqrt(n_folds)
rmse_yhats
rmse_yhats_mean
rmse_yhats_sem

rsq_yhats = np.array(rsq_yhats)
rsq_yhats_mean = np.mean(rsq_yhats)
rsq_yhats_sem = np.std(rsq_yhats)/math.sqrt(n_folds)
rsq_yhats
rsq_yhats_mean
rsq_yhats_sem

# ------ plot testing ------
y = np.concatenate([training_scaler_Y.inverse_transform(
    trainingY), training_scaler_Y.inverse_transform(testY)])
y_true = y.reshape(y.shape[0], )
yhats_training_mean_conv = yhats_training_mean_conv.reshape(
    yhats_training_mean_conv.shape[0], )
yhats_training_std = yhats_training_std.reshape(
    yhats_training_std.shape[0], )
yhats_training_sem = yhats_training_sem.reshape(
    yhats_training_sem.shape[0], )
yhats_test_mean_conv = yhats_test_mean_conv.reshape(
    yhats_test_mean_conv.shape[0], )
yhats_test_std = yhats_test_std.reshape(yhats_test_std.shape[0], )
yhats_test_sem = yhats_test_sem.reshape(yhats_test_sem.shape[0], )

y_yhat_plot(filepath=os.path.join(res_dir, 'new_lstm_phase1_base_scatter_'+freq+'.pdf'),
            y_true=y_true,
            training_yhat=yhats_training_mean_conv,
            training_yhat_err=yhats_training_std,
            test_yhat=yhats_test_mean_conv,
            test_yhat_err=yhats_test_std,
            plot_title='Cross-validation prediction',
            ylabel='PCL', xlabel='Subjects', plot_type='scatter',
            bar_width=0.25)


# ------ true test realm ------
