"""
to test small things
"""
import math
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import accuracy_score, mean_squared_error
# StratifiedKFold should be used for classification problems
# StratifiedKFold makes sure the fold has an equal representation of the classes
from sklearn.model_selection import KFold

from custom_functions.cv_functions import (NpArrayShapeError,
                                           PdDataFrameTypeError, idx_func,
                                           longitudinal_cv_xy_array,
                                           lstm_cv_train,
                                           lstm_ensemble_eval,
                                           lstm_ensemble_predict)
from custom_functions.data_processing import (inverse_norm_y,
                                              training_test_spliter)
from custom_functions.plot_functions import y_yhat_plot
from custom_functions.util_functions import logging_func


# ------ test functions ------
# NOTE: this should be a custom class
def lstm_cv(input, Y_colnames, remove_colnames, n_features,
            cv_n_folds=10,  cv_random_state=None,
            lstm_mode="simple"):
    """
    # Purpose:
        This is the main function for k-fold cross validation for LSTM RNN

    # Arguments:
        input
        Y_colnames
        remove_colnames
        cv_n_folds
        cv_random_state
        lstm_mode

    # Return
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
y = np.array(raw.loc[:, 'PCL'])

# ---- generate training and test sets with min-max normalization
training, test, scaler_X, scaler_Y = training_test_spliter(
    data=raw, training_percent=0.9, random_state=10, min_max_scaling=True,
    scale_column_as_y=['PCL'],
    scale_column_to_exclude=['subject', 'PCL', 'group'])

trainingX, trainingY = longitudinal_cv_xy_array(input=training, Y_colnames=['PCL'],
                                                remove_colnames=['subject', 'group'], n_features=8)
testX, testY = longitudinal_cv_xy_array(input=test, Y_colnames=['PCL'],
                                        remove_colnames=['subject', 'group'], n_features=8)


# below: as an example for converting the normalized Y back to measured values
# _, testY = inverse_norm_y(training_y=trainingY, test_y=testY, scaler=scaler_Y)

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
    print('fold: ', fold_id)
    cv_train_X, cv_train_Y = X[cv_train_idx[i]], Y[cv_train_idx[i]]
    cv_test_X, cv_test_Y = X[cv_test_idx[i]], Y[cv_test_idx[i]]
    cv_m, cv_m_history, cv_m_test_rmse = lstm_cv_train(trainX=cv_train_X, trainY=cv_train_Y,
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
np.std(cv_m_test_rmse_ensemble)  # 0.112
np.mean(cv_m_test_rmse_ensemble)  # 0.298

# ------ prediction ------
# prediction for 20% test subjests
yhats_testX = lstm_ensemble_predict(
    models=cv_m_ensemble, n_members=len(cv_m_ensemble), testX=testX)
# below: anix=0 means "along the row, by column"
yhats_testX_mean = np.mean(yhats_testX, axis=0)
yhats_testX_std = np.std(yhats_testX, axis=0)
yhats_testX_sem = yhats_testX_std/math.sqrt(n_folds)

# prediction for 80% test subjests
yhats_trainingX = lstm_ensemble_predict(
    models=cv_m_ensemble, n_members=len(cv_m_ensemble), testX=trainingX)
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
trainingX_conv, testY_conv = inverse_norm_y(training_y=trainingY,
                                            test_y=testY, scaler=scaler_Y)

# ------ eval ------
test_rmse = lstm_ensemble_eval(models=cv_m_ensemble, n_members=len(cv_m_ensemble),
                               testX=testX, testY=testY)
test_rmse = np.array(test_rmse)
np.std(test_rmse)  # 0.074
np.mean(test_rmse)  # 0.365

# ------ plot testing ------
y = np.concatenate([trainingY, testY])
y_true = scaler_Y.inverse_transform(y.reshape(y.shape[0], 1))
y_true = y_true.reshape(y_true.shape[0], )

y_yhat_plot(filepath=os.path.join(res_dir, 'cv_plot_test.pdf'),
            y_true=y_true,
            training_yhat=yhats_trainingX_pred,
            training_yhat_err=yhats_trainingX_sem,
            test_yhat=yhats_testX_pred,
            test_yhat_err=yhats_testX_sem,
            plot_title='CV RMSE',
            ylabel='PCL', xlabel='Subjects', plot_type='scatter',
            bar_width=0.25)

# ------ test OO syntax for plotting ------
filepath = os.path.join(res_dir, 'cv_plot_test.pdf')
y_true = y_true
training_yhat = yhats_trainingX_pred
training_yhat_err = yhats_trainingX_sem
test_yhat = yhats_testX_pred
test_yhat_err = yhats_testX_sem
plot_title = 'CV RMSE'
ylabel = 'PCL'
xlabel = 'Subjects'
plot_type = 'bar'
bar_width = 0.25

y = y_true
x = np.arange(1, len(y)+1)

# ---- one plot
training_yhat_plot, training_yhat_err_plot = np.empty_like(y), np.empty_like(y)
training_yhat_plot[:, ], training_yhat_err_plot[:, ] = np.nan, np.nan
training_yhat_plot[0:training_yhat.shape[0],
                   ], training_yhat_err_plot[0:training_yhat_err.shape[0], ] = training_yhat, training_yhat_err

test_yhat_plot, test_yhat_err_plot = np.empty_like(y), np.empty_like(y)
test_yhat_plot[:, ], test_yhat_err_plot[:, ] = np.nan, np.nan
test_yhat_plot[training_yhat.shape[0]:,
               ], test_yhat_err_plot[training_yhat_err.shape[0]:, ] = test_yhat, test_yhat_err

# distance
r1 = np.arange(1, len(y)+1) - bar_width/2
r2 = np.arange(1, len(y)+1) + bar_width/2
r3 = r2

# OO syntax
fig, ax = plt.subplots(figsize=(9, 3))
ax.set_xlim((0, 33))
fig.set_facecolor('white')
ax.set_facecolor('white')
ax.bar(r1, y, width=bar_width, color='red', label='original')
ax.bar(r2, training_yhat_plot, yerr=training_yhat_err_plot,
       width=bar_width, color='gray', label='training', ecolor='black', capsize=0)
ax.bar(r3, test_yhat_plot, yerr=test_yhat_err_plot,
       width=bar_width, color='blue', label='test', ecolor='black', capsize=0)
ax.axhline(color='black')
ax.set_title(plot_title, color='black')
ax.set_xlabel(xlabel, fontsize=10, color='black')
ax.set_ylabel(ylabel, fontsize=10, color='black')
ax.tick_params(labelsize=5, color='black', labelcolor='black')
plt.setp(ax.spines.values(), color='black')
leg = ax.legend(loc='best', ncol=3, fontsize=8, facecolor='white')
for text in leg.get_texts():
    text.set_color('black')
    text.set_weight('bold')
    text.set_alpha(0.5)

fig.savefig(filepath, dpi=600, bbox_inches='tight', facecolor='white')
fig


fig, ax = plt.subplots(figsize=(9, 3), facecolor='white')
ax.set_xlim((0, 33))
fig.set_facecolor('white')
ax.set_facecolor('white')
ax.scatter(x, y, color='red', label='original')
ax.fill_between(x, training_yhat_plot-training_yhat_err_plot,
                training_yhat_plot+training_yhat_err_plot, color='gray', alpha=0.2,
                label='training')
ax.fill_between(x, test_yhat_plot-test_yhat_err_plot,
                test_yhat_plot+test_yhat_err_plot, color='blue', alpha=0.2,
                label='test')
ax.set_title(plot_title, color='black')
ax.set_xlabel(xlabel, fontsize=10, color='black')
ax.set_ylabel(ylabel, fontsize=10, color='black')
ax.tick_params(labelsize=5, color='black', labelcolor='black')
plt.setp(ax.spines.values(), color='black')
leg = ax.legend(loc='best', ncol=3, fontsize=8, facecolor='white')
for text in leg.get_texts():
    text.set_color('black')
    text.set_weight('bold')
    text.set_alpha(0.5)
fig.savefig(filepath, dpi=600, bbox_inches='tight', facecolor='white')
fig
