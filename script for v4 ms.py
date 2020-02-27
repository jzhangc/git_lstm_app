"""
NEW! phase1 based single timepoint to phase1-phase2 lstm analysis
"""
# from sklearn.preprocessing import MinMaxScaler, StandardScaler

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

# ---- import data
freq = 2
# raw = pd.read_csv(os.path.join(
#     dat_dir, 'lstm_aec_phases_freq2.csv'), engine='python')
raw = pd.read_csv(os.path.join(
    dat_dir, 'lstm_aec_phases_freq'+str(freq)+'_v4.csv'), engine='python')

raw.iloc[0:5, 0:5]
raw.shape

# ---- key variable
n_features = 10
n_folds = 10
selected_features = [['PP19', 'PN14', 'PN05'],
                     ['PN08', 'PP19', 'PN21'],
                     ['PN10', 'PP19', 'PN16'],
                     ['PN05', 'PN20', 'PP10'],
                     ['PN19', 'PP15', 'PN13'],
                     ['PN10', 'PP18', 'PN14'],
                     ['PN27', 'PN18', 'PP22']]  # order: freq 1~7

# result directory
res_dir = os.path.join(main_dir, 'results', 'freq'+str(freq))


# ---- generate training and test sets with min-max normalization
# new spliting
training, test, _, _ = training_test_spliter_final(
    data=raw, random_state=1,
    man_split=True, man_split_colname='subject', man_split_testset_value=selected_features[freq-1],
    x_standardization=False,
    y_min_max_scaling=False)

# below: as an example for converting the normalized Y back to measured values
# _, testY = inverse_norm_y(training_y=trainingY, test_y=testY, scaler=scaler_Y)
# ---- test k-fold data sampling
cv_train_idx, cv_test_idx = idx_func(input=training, n_features=n_features, Y_colnames=['PCL'],
                                     remove_colnames=['subject', 'group'], n_folds=n_folds, random_state=13)

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
                                                                               hidden_units=n_features*5, epochs=150, batch_size=29,
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
                           'freq'+str(freq)+'_cv_'+'fold_'+str(i+1)+'.h5'))
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

# ------ prediction ------
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


# also features: inverse the predictions from normalized values to PCLs
# prediction for the test subjests
yhats_test = lstm_ensemble_predict(testX=testX,
                                   models=cv_m_ensemble, model_index=top5_index)
# yhats_test = lstm_ensemble_predict(testX=testX, models=cv_m_ensemble)
# below: anix=0 means "along the row, by column"
# NOTE: below: real world utility only uses training scalers
yhats_test_conv = [training_scaler_Y.inverse_transform(
    ary) for ary in yhats_test]
yhats_test_conv = np.array(yhats_test_conv)
yhats_test_mean_conv = np.mean(yhats_test_conv, axis=0)
yhats_test_std = np.std(yhats_test_conv, axis=0)
yhats_test_sem = yhats_test_std/math.sqrt(5)

# prediction for 80% test subjests
yhats_training = lstm_ensemble_predict(
    models=cv_m_ensemble, testX=trainingX, model_index=top5_index)
# yhats_training = lstm_ensemble_predict(
#     models=cv_m_ensemble, testX=trainingX)
yhats_training_conv = [training_scaler_Y.inverse_transform(
    ary) for ary in yhats_training]
yhats_training_conv = np.array(yhats_training_conv)
yhats_training_mean_conv = np.mean(yhats_training_conv, axis=0)
yhats_training_std = np.std(yhats_training_conv, axis=0)
yhats_training_sem = yhats_training_std/math.sqrt(5)

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
rmse_yhats_sd = np.std(rmse_yhats)
rmse_yhats_sem = np.std(rmse_yhats)/math.sqrt(5)
rmse_yhats
rmse_yhats_mean
rmse_yhats_sd
rmse_yhats_sem

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

y_yhat_plot(filepath=os.path.join(res_dir, 'new_lstm_phase1_base_scatter_freq'+str(freq)+'.pdf'),
            y_true=y_true,
            training_yhat=yhats_training_mean_conv,
            training_yhat_err=yhats_training_std,
            test_yhat=yhats_test_mean_conv,
            test_yhat_err=yhats_test_std,
            plot_title='Cross-validation prediction',
            ylabel='PCL', xlabel='Subjects', plot_type='scatter',
            bar_width=0.25)


# ------ true test realm ------
