"""
NOTE: old version: normalization after splitting (and odd way of calculating standard deviation and SEM)
NOTE: this won't work unless moved out of the old folder
"""
import math
import os

import numpy as np
import pandas as pd
# from keras.callbacks import History  # for input argument type check
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
# StratifiedKFold should be used for classification problems
# StratifiedKFold makes sure the fold has an equal representation of the classes
# from sklearn.model_selection import KFold
from custom_functions.custom_exceptions import (NpArrayShapeError,
                                                PdDataFrameTypeError)
from custom_functions.cv_functions import (idx_func, longitudinal_cv_xy_array,
                                           lstm_cv_train, lstm_ensemble_eval,
                                           lstm_ensemble_predict)
from custom_functions.data_processing import (inverse_norm_y,
                                              training_test_spliter)
from custom_functions.plot_functions import y_yhat_plot
from custom_functions.util_functions import logging_func


# ------ test functions ------


# ------ data processing ------
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
    dat_dir, 'lstm_aec_phases_freq6_new.csv'), engine='python')
raw.iloc[0:5, 0:5]
y = np.array(raw.loc[:, 'PCL'])

# ---- key variable
n_features = 12
n_folds = 10

# ---- generate training and test sets with min-max normalization
training, test, scaler_X, scaler_Y = training_test_spliter(
    data=raw, training_percent=0.9, random_state=10, min_max_scaling=True,
    scale_column_as_y=['PCL'],
    scale_column_to_exclude=['subject', 'PCL', 'group'])

# -- optional: fixed training/test split --
# theta
training, test, scaler_X, scaler_Y = training_test_spliter(
    data=raw, man_split=True, man_split_colname='subject', man_split_testset_value=['PN14', 'PN27', 'PP13'],
    min_max_scaling=True,
    scale_column_as_y=['PCL'],
    scale_column_to_exclude=['subject', 'PCL', 'group'])

# alpha
training, test, scaler_X, scaler_Y = training_test_spliter(
    data=raw, man_split=True, man_split_colname='subject', man_split_testset_value=['PN10', 'PN27', 'PP10'],
    min_max_scaling=True,
    scale_column_as_y=['PCL'],
    scale_column_to_exclude=['subject', 'PCL', 'group'])

# beta
training, test, scaler_X, scaler_Y = training_test_spliter(
    data=raw, man_split=True, man_split_colname='subject', man_split_testset_value=['PN09', 'PN14', 'PP20'],
    min_max_scaling=True,
    scale_column_as_y=['PCL'],
    scale_column_to_exclude=['subject', 'PCL', 'group'])

# low gamma one
training, test, scaler_X, scaler_Y = training_test_spliter(
    data=raw, man_split=True, man_split_colname='subject', man_split_testset_value=['PN11', 'PP12', 'PP15'],
    min_max_scaling=True,
    scale_column_as_y=['PCL'],
    scale_column_to_exclude=['subject', 'PCL', 'group'])

# low gamma two
training, test, scaler_X, scaler_Y = training_test_spliter(
    data=raw, man_split=True, man_split_colname='subject', man_split_testset_value=['PN08', 'PP12', 'PP18'],
    min_max_scaling=True,
    scale_column_as_y=['PCL'],
    scale_column_to_exclude=['subject', 'PCL', 'group'])

# low gamma three
training, test, scaler_X, scaler_Y = training_test_spliter(
    data=raw, man_split=True, man_split_colname='subject', man_split_testset_value=['PN09', 'PN17', 'PP21'],
    min_max_scaling=True,
    scale_column_as_y=['PCL'],
    scale_column_to_exclude=['subject', 'PCL', 'group'])

# high gamma
training, test, scaler_X, scaler_Y = training_test_spliter(
    data=raw, man_split=True, man_split_colname='subject', man_split_testset_value=['PN10', 'PN19', 'PP03'],
    min_max_scaling=True,
    scale_column_as_y=['PCL'],
    scale_column_to_exclude=['subject', 'PCL', 'group'])


# procesing
trainingX, trainingY = longitudinal_cv_xy_array(input=training, Y_colnames=['PCL'],
                                                remove_colnames=['subject', 'group'], n_features=n_features)
testX, testY = longitudinal_cv_xy_array(input=test, Y_colnames=['PCL'],
                                        remove_colnames=['subject', 'group'], n_features=n_features)

# below: as an example for converting the normalized Y back to measured values
# _, testY = inverse_norm_y(training_y=trainingY, test_y=testY, scaler=scaler_Y)
# ---- test k-fold data sampling
cv_train_idx, cv_test_idx = idx_func(input=training, n_features=n_features, Y_colnames=['PCL'],
                                     remove_colnames=['subject', 'group'], n_folds=n_folds, random_state=21)

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
                                                                      output_activation='sigmoid',
                                                                      hidden_units=6, epochs=300, batch_size=29,
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
cv_rsq_mean = np.mean(cv_m_test_rsq_ensemble)
cv_rsq_sem = np.std(cv_m_test_rsq_ensemble)/math.sqrt(n_folds)
cv_rmse_mean
cv_rmse_sem
cv_m_test_rmse_ensemble
cv_m_test_rsq_ensemble
cv_rsq_mean
cv_rsq_sem

# ------ prediction ------
# prediction for 20% test subjests
yhats_testX = lstm_ensemble_predict(
    models=cv_m_ensemble,  testX=testX)
# below: anix=0 means "along the row, by column"
yhats_testX_mean = np.mean(yhats_testX, axis=0)
yhats_testX_std = np.std(yhats_testX, axis=0)
yhats_testX_sem = yhats_testX_std/math.sqrt(n_folds)

# prediction for 80% test subjests
yhats_trainingX = lstm_ensemble_predict(
    models=cv_m_ensemble, testX=trainingX)
yhats_trainingX_mean = np.mean(yhats_trainingX, axis=0)
yhats_trainingX_std = np.std(yhats_trainingX, axis=0)
yhats_trainingX_sem = yhats_trainingX_std/math.sqrt(n_folds)

# inverse the predictions from normalized values to PCLs
yhats_trainingX_pred, yhats_testX_pred = inverse_norm_y(
    training_y=yhats_trainingX_mean, test_y=yhats_testX_mean, scaler=scaler_Y)
yhats_trainingX_std, yhats_testX_std = inverse_norm_y(
    training_y=yhats_trainingX_std, test_y=yhats_testX_std, scaler=scaler_Y)
yhats_trainingX_sem, yhats_testX_sem = inverse_norm_y(
    training_y=yhats_trainingX_sem, test_y=yhats_testX_sem, scaler=scaler_Y)
trainingY_conv, testY_conv = inverse_norm_y(training_y=trainingY,
                                            test_y=testY, scaler=scaler_Y)

# ------ eval ------
# test_rmse = lstm_ensemble_eval(models=cv_m_ensemble, n_members=len(cv_m_ensemble),
#                                testX=testX, testY=testY)
# test_rmse = np.array(test_rmse)
# test_rmse_mean = np.mean(test_rmse)  # 0.227
# test_rmse_sem = np.std(test_rmse)/math.sqrt(n_folds)

# below: calcuate RMSE and R2 (final, i.e. on the test data) using inversed y and yhat
# this RMSE is the score to report
rmse_yhats, rsq_yhats = list(), list()
for yhat_testX in yhats_testX:
    _, yhat_testX_conv = inverse_norm_y(
        training_y=yhats_trainingX_mean, test_y=yhat_testX, scaler=scaler_Y)
    rmse_yhat = math.sqrt(mean_squared_error(
        y_true=testY_conv, y_pred=yhat_testX_conv))
    rsq_yhat = r2_score(y_true=testY_conv, y_pred=yhat_testX_conv)
    rmse_yhats.append(rmse_yhat)
    rsq_yhats.append(rsq_yhat)
rmse_yhats = np.array(rmse_yhats)


rmse_yhats_mean = np.mean(rmse_yhats)
rmse_yhats_sem = np.std(rmse_yhats)/math.sqrt(n_folds)
rmse_yhats_mean
rmse_yhats_sem

rsq_yhats = np.array(rsq_yhats)
rsq_yhats_mean = np.mean(rsq_yhats)
rsq_yhats_sem = np.std(rsq_yhats)/math.sqrt(n_folds)
rsq_yhats
rsq_yhats_mean
rsq_yhats_sem

# ------ plot testing ------
y = np.concatenate([trainingY, testY])
y_true = scaler_Y.inverse_transform(y.reshape(y.shape[0], 1))
y_true = y_true.reshape(y_true.shape[0], )

y_yhat_plot(filepath=os.path.join(res_dir, 'oldtest_freq2_cv_plot_scatter.pdf'),
            y_true=y_true,
            training_yhat=yhats_trainingX_pred,
            training_yhat_err=yhats_trainingX_sem,
            test_yhat=yhats_testX_pred,
            test_yhat_err=yhats_testX_sem,
            plot_title='Cross-validation prediction',
            ylabel='PCL', xlabel='Subjects', plot_type='scatter',
            bar_width=0.25)


# ------ true test realm ------
tst_test = np.array([scaler_Y.inverse_transform(ary) for ary in yhats_testX])
tst_test_mean = np.mean(tst_test, axis=0)
tst_test_mean = tst_test_mean.reshape(tst_test_mean.shape[0], )
tst_test_std = np.std(tst_test, axis=0)
tst_test_std = tst_test_std.reshape(tst_test_std.shape[0], )


tst_training = np.array([scaler_Y.inverse_transform(ary)
                         for ary in yhats_trainingX])
tst_training_mean = np.mean(tst_training, axis=0)
tst_training_mean = tst_training_mean.reshape(tst_training_mean.shape[0], )
tst_training_std = np.std(tst_training, axis=0)
tst_training_std = tst_training_std.reshape(tst_training_std.shape[0], )

y_yhat_plot(filepath=os.path.join(res_dir, 'oldtest_freq6_cv_plot_scatter.pdf'),
            y_true=y_true,
            training_yhat=tst_training_mean,
            training_yhat_err=tst_training_std,
            test_yhat=tst_test_mean,
            test_yhat_err=tst_test_std,
            plot_title='Cross-validation prediction',
            ylabel='PCL', xlabel='Subjects', plot_type='scatter',
            bar_width=0.25)
